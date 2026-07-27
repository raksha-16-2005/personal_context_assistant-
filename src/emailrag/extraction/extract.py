"""Pulling commitments out of a message with an LLM.

The division of labour is the design: **the model reads, Python computes.** It
identifies who owes what to whom and quotes the deadline phrase verbatim; it is
never asked for a resolved date. Date arithmetic is where small models fail
confidently - returning a "Thursday" that falls on a Tuesday - and every such
error would have to be recomputed to be caught, so it is computed here instead
(see dates.py). This also makes the date-accuracy metric measure language
understanding rather than arithmetic.

Two arms, on purpose:

    qwen2.5:3b via Ollama    local, CPU-only here at ~25 s/message
    claude-haiku-4-5         hosted quality ceiling, ~$3 batched

The comparison is the result. A local model that reaches most of the ceiling's
accuracy means private mail never has to leave the machine, which is the whole
argument for phase 7; a local model that does not means the cost is a real choice
rather than a default. Rows from the two arms are never mixed - see
`schema.Commitment.model`.

Everything is cached (llm/cache.py). A re-run over the same 2-3k messages costs
nothing, which matters when one pass is an overnight job.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from ..llm.client import LLM, LLMError
from . import dates as D
from .schema import DIRECTIONS, KINDS, Commitment

MAX_BODY_CHARS = 4000

SYSTEM = """\
You extract commitments from corporate email. A commitment is a specific thing \
somebody owes somebody else: a deliverable, a meeting to attend, a payment, a \
review, a decision.

You return JSON and nothing else.

Critical rule about dates: quote the deadline exactly as the email words it, in \
`due_phrase`. Do NOT convert it to a calendar date, and do not compute anything. \
"next Thursday", "by EOD", "before the board meeting" are all correct values for \
`due_phrase`. A date will be computed from your phrase and the email's send date.

Do not invent commitments. A message that discusses something without anybody \
owing anything has no commitments, and an empty list is the right answer far more \
often than not. Pleasantries, FYIs, newsletters and forwarded articles have none."""

PROMPT = """\
Email:
From: {sender}
To: {recipients}
Date sent: {sent}
Subject: {subject}

{body}

---
Extract every commitment. Return JSON only:

{{"commitments": [
  {{"text": "what is owed, one short clause",
    "kind": "deliverable|meeting|payment|review|decision|other",
    "direction": "i_owe|they_owe|mutual|unclear",
    "owner": "who owes it - email address or name as written, else \\"\\"",
    "counterparty": "who is owed, else \\"\\"",
    "due_phrase": "the deadline exactly as worded, or \\"\\" if none is stated",
    "confidence": 0.0-1.0,
    "quote": "the sentence you took it from"}}
]}}

If there are no commitments, return {{"commitments": []}}."""


@dataclass
class ExtractionStats:
    messages_seen: int = 0
    messages_prefiltered_out: int = 0
    messages_called: int = 0
    commitments: int = 0
    with_due_date: int = 0
    unresolvable_phrases: int = 0
    ambiguous_dates: int = 0
    rolled_year: int = 0
    invalid_rows: int = 0
    failures: int = 0
    cached_calls: int = 0
    seconds: float = 0.0

    @property
    def prefilter_pass_rate(self) -> float:
        return (self.messages_called / self.messages_seen) if self.messages_seen else 0.0

    def render(self) -> str:
        lines = [
            f"messages            {self.messages_seen:,}",
            f"  prefiltered out   {self.messages_prefiltered_out:,} "
            f"({1 - self.prefilter_pass_rate:.0%})",
            f"  sent to the model {self.messages_called:,}",
            f"commitments         {self.commitments:,}",
            f"  with a due date   {self.with_due_date:,}",
            f"  ambiguous date    {self.ambiguous_dates:,}",
            f"  year rolled       {self.rolled_year:,}",
            f"  phrase unresolved {self.unresolvable_phrases:,}",
            f"invalid rows        {self.invalid_rows:,}",
            f"extraction failures {self.failures:,}",
            f"cached calls        {self.cached_calls:,}",
        ]
        if self.messages_called:
            lines.append(f"seconds/message     {self.seconds / self.messages_called:.1f}")
        return "\n".join(lines)


class CommitmentExtractor:
    def __init__(self, llm: LLM | None = None, *, prefilter: bool = True,
                 max_body_chars: int = MAX_BODY_CHARS) -> None:
        self._llm = llm
        self.prefilter = prefilter
        self.max_body_chars = max_body_chars
        self.stats = ExtractionStats()
        self.errors: list[dict] = []

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM("ollama")
        return self._llm

    @property
    def model_name(self) -> str:
        return getattr(self.llm, "model", "unknown")

    def extract(self, message: dict) -> list[Commitment]:
        """Commitments in one message. Never raises - a failed message is
        recorded and skipped, because one pass is an overnight job."""
        self.stats.messages_seen += 1
        body = (message.get("body_new") or "")[:self.max_body_chars]
        subject = message.get("subject") or ""

        if self.prefilter and not D.looks_like_commitment(f"{subject}\n{body}"):
            self.stats.messages_prefiltered_out += 1
            return []

        sent = message.get("date_utc")
        prompt = PROMPT.format(
            sender=message.get("sender") or "unknown",
            recipients=(message.get("recipients") or "unknown")[:200],
            sent=_sent_label(sent),
            subject=subject,
            body=body or "(empty body)",
        )

        self.stats.messages_called += 1
        t0 = time.perf_counter()
        try:
            data = self.llm.json_complete(prompt, system=SYSTEM, max_tokens=1200)
        except (LLMError, json.JSONDecodeError) as exc:
            self.stats.failures += 1
            self.errors.append({"message_id": message.get("dedup_key"),
                                "error": f"{type(exc).__name__}: {exc}"})
            return []
        finally:
            self.stats.seconds += time.perf_counter() - t0
        if getattr(self.llm, "last_cached", False):
            self.stats.cached_calls += 1

        return self._to_commitments(data, message, sent)

    def _to_commitments(self, data: object, message: dict, sent) -> list[Commitment]:
        raw_items = data.get("commitments") if isinstance(data, dict) else data
        if not isinstance(raw_items, list):
            self.stats.failures += 1
            self.errors.append({"message_id": message.get("dedup_key"),
                                "error": f"expected a list, got {type(raw_items).__name__}"})
            return []

        out: list[Commitment] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            commitment = self._one(item, message, sent)
            if commitment is None:
                continue
            out.append(commitment)
        self.stats.commitments += len(out)
        return out

    def _one(self, item: dict, message: dict, sent) -> Commitment | None:
        from .schema import validate

        text = str(item.get("text") or "").strip()
        if not text:
            return None

        kind = str(item.get("kind") or "other").strip().lower()
        direction = str(item.get("direction") or "unclear").strip().lower()
        c = Commitment(
            message_id=message.get("dedup_key") or "",
            text=text[:500],
            # An out-of-vocabulary label is coerced rather than dropped: the
            # commitment is still real, and losing it to a taxonomy mismatch would
            # be a worse error than filing it as "other".
            kind=kind if kind in KINDS else "other",
            direction=direction if direction in DIRECTIONS else "unclear",
            owner=str(item.get("owner") or "")[:200],
            counterparty=str(item.get("counterparty") or "")[:200],
            due_phrase=str(item.get("due_phrase") or "").strip()[:200],
            confidence=_clamp(item.get("confidence")),
            quote=str(item.get("quote") or "")[:600],
            model=self.model_name,
            extracted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        if c.due_phrase and sent is not None:
            resolved = D.try_resolve(c.due_phrase, sent)
            if resolved is None:
                # A phrase like "before the board meeting" is a real deadline with
                # no computable date. Kept, with the phrase, and counted.
                self.stats.unresolvable_phrases += 1
            else:
                c.due_at = resolved.value
                c.due_precision = resolved.precision
                c.due_ambiguous = resolved.ambiguous
                c.due_alternative = resolved.alternative
                c.due_rolled_year = resolved.rolled_year
                self.stats.with_due_date += 1
                self.stats.ambiguous_dates += int(resolved.ambiguous)
                self.stats.rolled_year += int(resolved.rolled_year)

        problems = validate(c)
        if problems:
            self.stats.invalid_rows += 1
            self.errors.append({"message_id": c.message_id, "error": "; ".join(problems)})
            return None
        return c


def _sent_label(sent) -> str:
    """The send date, with its weekday spelled out.

    The weekday is not decoration: "next Thursday" cannot be resolved without
    knowing what day the message was sent, and stating it lets the model reason
    about whether a phrase is even a deadline. The resolution itself still happens
    in Python.
    """
    if sent is None:
        return "unknown"
    if isinstance(sent, datetime):
        sent = sent.date()
    if isinstance(sent, date):
        return f"{sent.isoformat()} ({sent.strftime('%A')})"
    return str(sent)


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))       # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
