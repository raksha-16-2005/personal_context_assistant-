"""Parse the CMU Enron maildir into a normalized message table.

The corpus is ~517k files across ~150 user directories, but the true number of
distinct messages is far lower: every message appears once in the sender's
``sent`` folder and once in each recipient's folder, and many users kept
``all_documents`` mirrors of their own inbox. Deduplication - not the bulk-mail
filter - is what actually shrinks this corpus.

Output is a Parquet dataset written in batches so peak RSS stays flat; the
whole parse fits comfortably in the 16 GB budget.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email import policy
from email.header import decode_header
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Iterator

# Enron bodies are a mix of ASCII, cp1252 (smart quotes pasted from Word) and
# occasional latin-1. Try in order of specificity; the last never fails.
_DECODE_CHAIN = ("utf-8", "cp1252", "latin-1")

# Markers that begin quoted history in Outlook-era mail. Everything from the
# first match onward is prior-message content, not new text from this sender.
_QUOTE_MARKERS = re.compile(
    r"""^(?:
          \s*-{2,}\s*Original\s+Message\s*-{2,}
        | \s*-{2,}\s*Forwarded\s+by\b.*
        | \s*_{10,}\s*$
        | \s*From:\s.+\n\s*Sent:\s.+
        | On\s.{0,80}\swrote:\s*$
    )""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

_SUBJECT_PREFIX = re.compile(r"^\s*(?:(?:re|fw|fwd|aw|sv)\s*(?:\[\d+\])?\s*:\s*)+", re.IGNORECASE)


@dataclass(slots=True)
class Message:
    message_id: str
    dedup_key: str
    date_utc: datetime | None
    sender: str
    recipients: str          # ';'-joined; kept flat for Parquet friendliness
    cc: str
    subject: str
    subject_norm: str
    body: str                # full body as sent, quoted history included
    body_new: str            # quoted history stripped - what this sender wrote
    in_reply_to: str
    references: str
    has_list_unsubscribe: bool
    source_path: str
    owner: str               # maildir/<owner>/... - who this copy belonged to
    folder: str


def _decode(raw: bytes) -> str:
    for enc in _DECODE_CHAIN:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _hdr(msg, name: str) -> str:
    """Read a header as a plain, decoded str.

    Under compat32, `msg.get()` usually returns a str - but for a header
    containing RFC 2047 encoded words (`=?Windows-1252?Q?...?=`) it returns an
    `email.header.Header` instead. Two problems, not one: that object has no
    string methods (Enron has enough of those to crash a full parse), and
    `str()` on it - the obvious fix - returns the encoded form verbatim,
    since compat32 never decodes headers on its own. Every header read goes
    through `decode_header` so a real subject shows up instead of literal
    `=?...?=` gibberish in citations.
    """
    value = msg.get(name)
    if value is None:
        return ""
    raw = value if isinstance(value, str) else str(value)
    try:
        parts = decode_header(raw)
    except (UnicodeDecodeError, LookupError, ValueError):
        return raw
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _addr_list(value: str | None) -> str:
    if not value:
        return ""
    seen: list[str] = []
    for _, addr in getaddresses([value.replace("\n", " ")]):
        addr = addr.strip().lower()
        if addr and addr not in seen:
            seen.append(addr)
    return ";".join(seen)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    # A handful of Enron headers carry no offset; treat those as UTC rather
    # than emitting naive timestamps that break comparisons downstream.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _strip_quoted(body: str) -> str:
    """Return only the text this sender actually typed."""
    match = _QUOTE_MARKERS.search(body)
    if match:
        body = body[: match.start()]
    # Drop residual '>' quoting that survives when there is no banner marker.
    kept = [ln for ln in body.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(kept).strip()


def _normalize_subject(subject: str) -> str:
    return _SUBJECT_PREFIX.sub("", subject or "").strip().lower()


def _body_of(msg) -> str:
    """Flatten to text/plain. Enron is overwhelmingly single-part plain text."""
    if msg.is_multipart():
        parts = [
            _decode(p.get_payload(decode=True) or b"")
            for p in msg.walk()
            if p.get_content_type() == "text/plain"
        ]
        return "\n".join(parts)
    return _decode(msg.get_payload(decode=True) or b"")


def parse_message(raw: bytes, source_path: str = "", owner: str = "",
                  folder: str = "") -> Message | None:
    """Parse RFC822 bytes into a `Message`.

    Split out from `parse_file` so Gmail can use it. Gmail's API returns RFC822
    with `format=raw`, so both corpora go through exactly this function - which is
    what makes any difference in their numbers a difference in the mail rather than
    in the code. A second parser for the second corpus would quietly produce
    incomparable rows: different body extraction, different address normalisation,
    a different dedup key for the same message.
    """
    try:
        msg = BytesParser(policy=policy.compat32).parsebytes(raw)
    except (ValueError, TypeError):
        return None

    body = _body_of(msg)
    subject = _hdr(msg, "Subject").replace("\n", " ").strip()
    sender = _addr_list(_hdr(msg, "From"))
    date_utc = _parse_date(_hdr(msg, "Date"))

    message_id = _hdr(msg, "Message-ID").strip()
    recipients = _addr_list(_hdr(msg, "To"))
    cc = _addr_list(_hdr(msg, "Cc"))

    # Identity is CONTENT, not Message-ID.
    #
    # The obvious key is Message-ID, and on this corpus it is worthless: the
    # JavaMail export stamped a fresh Message-ID on every copy, so all 435,498
    # surviving files have distinct ids and ID-based dedup removes exactly
    # nothing. Hashing the identity-bearing fields instead collapses 49.8% of
    # the corpus - the sender's Sent copy and every recipient's filed copy of
    # one message become one row.
    #
    # Recipients and cc are inside the hash on purpose. Copies of a single
    # message share them; two genuinely separate sends of the same text to
    # different people do not, and must not be merged - entity-scoped queries
    # depend on that distinction.
    digest = hashlib.sha256("\x00".join([
        sender, recipients, cc, subject, str(date_utc), body[:8000],
    ]).encode("utf-8", "replace")).hexdigest()
    dedup_key = f"sha256:{digest}"

    return Message(
        message_id=message_id,
        dedup_key=dedup_key,
        date_utc=date_utc,
        sender=sender,
        recipients=recipients,
        cc=cc,
        subject=subject,
        subject_norm=_normalize_subject(subject),
        body=body,
        body_new=_strip_quoted(body),
        in_reply_to=_hdr(msg, "In-Reply-To").strip(),
        references=_hdr(msg, "References").strip(),
        has_list_unsubscribe=msg.get("List-Unsubscribe") is not None,
        source_path=source_path,
        owner=owner,
        folder=folder,
    )


def parse_file(path: Path, maildir_root: Path) -> Message | None:
    """Parse one maildir file. Owner and folder come from its path."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    try:
        rel = path.relative_to(maildir_root)
        owner = rel.parts[0]
        folder = "/".join(rel.parts[1:-1])
        source_path = str(rel)
    except ValueError:
        owner, folder, source_path = "", "", str(path)

    return parse_message(raw, source_path=source_path, owner=owner, folder=folder)


def iter_user_dirs(maildir_root: Path) -> list[Path]:
    return sorted(p for p in maildir_root.iterdir() if p.is_dir())


def parse_user_dir(args: tuple[Path, Path]) -> list[dict]:
    """Worker entry point - one maildir user per task."""
    user_dir, maildir_root = args
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(user_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            msg = parse_file(Path(dirpath) / fn, maildir_root)
            if msg is not None:
                out.append(asdict(msg))
    return out


def iter_all_files(maildir_root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(maildir_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.startswith("."):
                yield Path(dirpath) / fn
