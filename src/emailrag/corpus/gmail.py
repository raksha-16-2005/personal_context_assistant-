"""Gmail as a corpus source, producing the same rows `enron.py` does.

The pipeline is corpus-agnostic by design and this is the payoff: Gmail's API
returns RFC822 messages, which is exactly what `enron.py` already parses. So this
module is an *acquisition* layer, not a second parser - it fetches raw bytes and
hands them to the existing parser, which means every downstream stage (dedup, bulk
filter, threading, chunking, retrieval) behaves identically on both corpora and any
difference in the numbers is a difference in the mail, not in the code.

Three things that will bite anyone using this, documented because they are
properties of Google's platform rather than of this code:

**Refresh tokens expire after 7 days while the OAuth app is in Testing.** The
long-lived-token exception covers basic profile scopes only, not Gmail. Escaping it
needs Google app verification, which for a Gmail scope means a security review. For
personal use that means re-authorising weekly, and `TokenStore` reports how long is
left rather than failing opaquely on day eight.

**`gmail.readonly` is the whole mailbox.** There is no narrower scope that still
allows search. The token this stores can read everything, so it lives in a
gitignored file with 0600 permissions and never in an environment variable that a
subprocess or a crash reporter could pick up.

**History-based sync is the only affordable re-run.** A full `messages.list` walk
re-enumerates the mailbox every time; `historyId` fetches only what changed. Google
expires history older than roughly a week, so the code must handle "your cursor is
too old" as a normal event and fall back to a full sync - not as an error.

Requests use urllib, matching `llm/client.py`: the Google API client library brings
a large dependency tree to a stack pinned tightly around torch 2.2.2, and these are
plain HTTPS calls with a bearer token.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://gmail.googleapis.com/gmail/v1"
TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# The narrowest scope that still allows reading mail. There is no read-only scope
# limited to a subset of the mailbox.
SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

# The web app's calendar-suggestions flow needs to create events once a user
# confirms one (see the plan's "suggest, then user confirms" decision) - a
# separate constant, not folded into SCOPE, because the desktop-app flow
# (scripts/gmail_auth.py) has no calendar feature and must keep asking for
# Gmail only.
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events"

# Google's documented behaviour for apps in Testing. Recorded here so the warning
# can be specific instead of "your token stopped working".
TESTING_REFRESH_TOKEN_DAYS = 7

DEFAULT_TOKEN_PATH = Path.home() / ".config" / "emailrag" / "gmail_token.json"

TIMEOUT = 60


class GmailError(RuntimeError):
    pass


class NeedsAuthorization(GmailError):
    """No usable token. Carries the exact steps, not a stack trace."""


class HistoryTooOld(GmailError):
    """The sync cursor is older than Google retains - fall back to a full sync."""


# -- tokens -----------------------------------------------------------------

class TokenStoreBase:
    """Token-freshness logic shared by every storage backend.

    Everything here reads or computes from `self.data` alone, so it is
    identical whichever backend holds the bytes. `TokenStore` below persists
    to a local JSON file, for the single-user desktop-app flow
    `scripts/gmail_auth.py` drives. The multi-tenant web app's
    `DbTokenStore` (`webapp/app/tokens/store.py`) persists an encrypted row
    per user in Postgres instead - same interface, so `GmailClient` needs no
    changes to run against either one.
    """

    data: dict

    @property
    def refresh_token(self) -> str:
        return self.data.get("refresh_token", "")

    @property
    def access_token(self) -> str:
        return self.data.get("access_token", "")

    @property
    def expires_at(self) -> float:
        return float(self.data.get("expires_at", 0))

    @property
    def issued_at(self) -> float:
        return float(self.data.get("issued_at", 0))

    def access_token_valid(self, skew: int = 60) -> bool:
        return bool(self.access_token) and time.time() < self.expires_at - skew

    def refresh_token_age_days(self) -> float:
        return (time.time() - self.issued_at) / 86400 if self.issued_at else 0.0

    def days_until_reauthorization(self) -> float:
        """How long before a Testing-status refresh token stops working.

        Reported rather than discovered: the failure otherwise arrives on day eight
        as an opaque `invalid_grant` in the middle of a sync.
        """
        return max(0.0, TESTING_REFRESH_TOKEN_DAYS - self.refresh_token_age_days())

    def warn_if_expiring(self) -> str:
        if not self.refresh_token:
            return ""
        left = self.days_until_reauthorization()
        if left <= 0:
            return ("This refresh token is older than 7 days. Apps in Testing get "
                    "7-day refresh tokens for Gmail scopes, so re-run the "
                    "authorization step.")
        if left <= 2:
            return (f"Refresh token expires in {left:.1f} days "
                    f"(Testing-status limit for Gmail scopes).")
        return ""


@dataclass
class TokenStore(TokenStoreBase):
    """OAuth tokens on disk, 0600, outside the repo.

    Outside the repo deliberately: `.gitignore` is a convention and a `git add -f`
    or a moved path defeats it. A token that can read an entire mailbox should not
    be one mistake away from a public commit, so it lives under ~/.config by
    default and the repo never has to be trusted with it.
    """

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    data: dict = field(default_factory=dict)

    def load(self) -> "TokenStore":
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written 0600 before any content lands in it, so the secret is never
        # briefly world-readable.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(self.data, fh, indent=2)


def authorization_url(client_id: str, redirect_uri: str = "http://localhost:8765/",
                      scopes: str = SCOPE, state: str = "") -> str:
    """The consent URL. `access_type=offline` is what yields a refresh token at
    all; without `prompt=consent` Google omits it on re-authorisation.

    `scopes` is a space-separated list of scope URIs, defaulting to Gmail-only
    for the desktop-app flow. The web app passes `f"{SCOPE} {CALENDAR_SCOPE}"`
    so both grants happen on one consent screen.

    `state`, when given, is echoed back on the callback unmodified - the web
    app's CSRF defence against a browser being tricked into completing login
    with an attacker-supplied authorization code. The desktop loopback flow
    (localhost-only, single machine) has never needed it, so it stays opt-in
    rather than mandatory.
    """
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "access_type": "offline",
        "prompt": "consent",
    }
    if state:
        params["state"] = state
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def _post_form(url: str, form: dict) -> dict:
    body = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        if "invalid_grant" in detail:
            raise NeedsAuthorization(
                "the refresh token was rejected (invalid_grant).\n"
                "  Apps in Testing status get 7-day refresh tokens for Gmail "
                "scopes - this is expected, not a bug.\n"
                "  Re-run: python scripts/gmail_auth.py"
            ) from exc
        raise GmailError(f"token endpoint HTTP {exc.code}: {detail}") from exc


def exchange_code(client_id: str, client_secret: str, code: str,
                  redirect_uri: str = "http://localhost:8765/") -> dict:
    return _post_form(TOKEN_URL, {
        "client_id": client_id, "client_secret": client_secret, "code": code,
        "grant_type": "authorization_code", "redirect_uri": redirect_uri,
    })


def refresh_access_token(client_id: str, client_secret: str,
                         refresh_token: str) -> dict:
    return _post_form(TOKEN_URL, {
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token",
    })


# -- the client -------------------------------------------------------------

@dataclass
class SyncState:
    """Where the last sync got to.

    `history_id` is the cursor. Stored beside the corpus rather than in the token
    file: it is not a secret, and losing it should cost a full re-sync rather than
    a re-authorisation.
    """

    history_id: str = ""
    last_sync_utc: str = ""
    messages_seen: int = 0

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "history_id": self.history_id,
            "last_sync_utc": self.last_sync_utc,
            "messages_seen": self.messages_seen,
        }, indent=2))


def ensure_access_token(client_id: str, client_secret: str,
                        tokens: TokenStoreBase) -> str:
    """A valid bearer token for `tokens`' account, refreshing and persisting
    if the cached access token has expired.

    One access token is valid for every scope its refresh token was granted -
    Gmail and Calendar alike - so this is not Gmail-specific despite living in
    this module; `GmailClient._bearer` and the web app's `CalendarClient`
    (`webapp/app/calendar/client.py`) both call it against the same
    `DbTokenStore` row rather than each refreshing (and re-persisting) their
    own copy.
    """
    if tokens.access_token_valid():
        return tokens.access_token
    if not tokens.refresh_token:
        raise NeedsAuthorization(
            "no refresh token stored.\n  run: python scripts/gmail_auth.py")

    fresh = refresh_access_token(client_id, client_secret, tokens.refresh_token)
    tokens.data.update({
        "access_token": fresh["access_token"],
        "expires_at": time.time() + int(fresh.get("expires_in", 3600)),
    })
    # issued_at tracks the *refresh* token's age, so it is only set when a new
    # refresh token arrives - overwriting it on every access-token refresh
    # would hide the 7-day clock entirely.
    if fresh.get("refresh_token"):
        tokens.data["refresh_token"] = fresh["refresh_token"]
        tokens.data["issued_at"] = time.time()
    tokens.save()
    return tokens.access_token


class GmailClient:
    def __init__(self, client_id: str, client_secret: str,
                 tokens: TokenStore | None = None) -> None:
        if not client_id or not client_secret:
            raise NeedsAuthorization(
                "GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET are not set.\n"
                "  1. Google Cloud console -> new project\n"
                "  2. Enable the Gmail API\n"
                "  3. OAuth consent screen -> External, status Testing, add "
                "yourself as a test user\n"
                "  4. Credentials -> OAuth client ID -> Desktop app\n"
                "  5. Put both values in .env (gitignored)\n"
                "  Then: python scripts/gmail_auth.py")
        self.client_id = client_id
        self.client_secret = client_secret
        self.tokens = (tokens or TokenStore()).load()
        # `fetch_messages` below calls `_bearer()` from many threads at once
        # against this one client/token pair. The lock only ever guards the
        # cheap "is the cached token still valid" check (and, rarely, the
        # actual refresh call) - the slow part, the per-message HTTP GET in
        # `_get`, happens after `_bearer()` has already returned, outside
        # this lock, so this does not serialize the concurrency it's meant
        # to allow.
        self._bearer_lock = threading.Lock()

    # -- auth ----------------------------------------------------------------

    def _bearer(self) -> str:
        with self._bearer_lock:
            return ensure_access_token(self.client_id, self.client_secret, self.tokens)

    def _get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        url = f"{API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self._bearer()}"})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:400]
                if exc.code == 404 and "historyId" in detail:
                    raise HistoryTooOld(
                        "the stored historyId is older than Google retains "
                        "(roughly a week) - a full sync is required") from exc
                if exc.code == 401:
                    # Force a refresh on the next attempt rather than failing: the
                    # access token may have expired mid-run.
                    self.tokens.data["expires_at"] = 0
                    last = GmailError(f"HTTP 401: {detail}")
                    continue
                if exc.code == 429 or exc.code >= 500:
                    last = GmailError(f"HTTP {exc.code}: {detail}")
                    time.sleep(2 ** attempt)
                    continue
                raise GmailError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = GmailError(f"network: {exc}")
                time.sleep(2 ** attempt)
        raise last or GmailError("exhausted retries")

    # -- reading -------------------------------------------------------------

    def profile(self) -> dict:
        return self._get("/users/me/profile")

    def list_message_ids(self, query: str = "", max_messages: int = 0,
                         include_spam_trash: bool = False):
        """Message ids matching `query`, following pagination.

        A generator: a mailbox can be 100k messages and materialising every id
        before fetching any is a long silence followed by a memory spike.
        """
        page_token = ""
        seen = 0
        while True:
            params = {"maxResults": 500, "includeSpamTrash": str(include_spam_trash).lower()}
            if query:
                params["q"] = query
            if page_token:
                params["pageToken"] = page_token
            data = self._get("/users/me/messages", params)

            for item in data.get("messages", []):
                yield item["id"]
                seen += 1
                if max_messages and seen >= max_messages:
                    return
            page_token = data.get("nextPageToken", "")
            if not page_token:
                return

    def raw_message(self, message_id: str) -> bytes:
        """One message as RFC822 bytes - exactly what `enron.py` parses.

        `format=raw` rather than the parsed `full` format on purpose: the parsed
        form would need a second header/body extractor, and two parsers over two
        corpora is how the two corpora start producing incomparable rows.
        """
        data = self._get(f"/users/me/messages/{message_id}", {"format": "raw"})
        raw = data.get("raw", "")
        if not raw:
            raise GmailError(f"message {message_id} returned no raw payload")
        # Gmail uses URL-safe base64 without padding.
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))

    def history_since(self, history_id: str):
        """Message ids added since `history_id`.

        Raises `HistoryTooOld` when the cursor has expired, which is a normal event
        after a week away and not an error - the caller falls back to a full sync.
        """
        page_token = ""
        while True:
            params = {"startHistoryId": history_id, "historyTypes": "messageAdded",
                      "maxResults": 500}
            if page_token:
                params["pageToken"] = page_token
            data = self._get("/users/me/history", params)

            for record in data.get("history", []):
                for added in record.get("messagesAdded", []):
                    message = added.get("message", {})
                    if message.get("id"):
                        yield message["id"]
            page_token = data.get("nextPageToken", "")
            if not page_token:
                return

    def current_history_id(self) -> str:
        return str(self.profile().get("historyId", ""))


# -- turning it into corpus rows -------------------------------------------

def since_query(days: int) -> str:
    """A Gmail search query for the recent window.

    Retrieval wants all history; extraction does not. Ollama is CPU-only at ~25 s a
    message, so a 35k-message mailbox is ~240 hours of extraction - capping it at a
    recent window is the difference between an overnight run and a season.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return f"after:{cutoff.strftime('%Y/%m/%d')}"


def before_query(days: int) -> str:
    """The complementary half of `since_query` - everything *older* than the
    recent window, for the multi-tenant web app's staged first sync
    (`webapp/app/ingestion/worker.py`'s `RECENT_SYNC_DAYS`/
    `backfill_history`): fetch the last `days` fast so a new user has
    something to chat with in about a minute, then backfill everything
    before that cutoff in the background without re-fetching what the fast
    pass already has.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return f"before:{cutoff.strftime('%Y/%m/%d')}"


# One `raw_message` call measured at ~500 ms against real Gmail - almost
# entirely spent waiting on the network, not on this process's CPU. Fetching
# serially therefore wastes nearly the entire wall clock idle; a mailbox of
# 9k messages measured at ~80 minutes serially and a few minutes at this
# concurrency. Gmail's per-user quota is 250 quota units/s and messages.get
# costs 5, so ~50 req/s is the hard ceiling - 20 workers at ~500 ms each is
# ~40 req/s, comfortably under it rather than riding the edge, since a 429
# there costs more in backoff (`GmailClient._get`'s `2**attempt` sleep) than
# the extra concurrency would have saved.
FETCH_WORKERS = 20


def fetch_messages(client: GmailClient, query: str = "", max_messages: int = 0,
                   on_progress=None, max_workers: int = FETCH_WORKERS) -> list[dict]:
    """Fetch and parse into the same `Message` rows `enron.py` produces.

    The parser is imported from `enron.py` rather than reimplemented: identical
    parsing is what makes any difference between the two corpora a difference in
    the mail rather than in the code.

    Fetched concurrently (see `FETCH_WORKERS`) - order is not preserved, which
    is fine: nothing downstream (dedup, threading, chunking) depends on the
    order messages arrive in.

    `on_progress(completed, total)`, if given, is called once immediately
    with `completed=0` (as soon as the exact total is known - this fully
    materializes `list_message_ids` before fetching anything, specifically
    so a caller has a real denominator to show progress against from the
    very start, not just once the first batch lands) and again every 50
    completions after that.
    """
    from dataclasses import asdict

    from .enron import parse_message

    def _fetch_one(message_id: str):
        raw = client.raw_message(message_id)
        return parse_message(raw, source_path=f"gmail:{message_id}")

    # Materialized rather than left as a generator: a thread pool needs every
    # id up front to submit work, and even a 100k-message mailbox is a
    # trivial amount of memory as bare id strings. This also happens to be
    # the exact count `on_progress` reports as `total` - not an estimate.
    message_ids = list(client.list_message_ids(query, max_messages))
    if on_progress:
        on_progress(0, len(message_ids))

    # Dicts, not `Message` objects: `parse_user_dir` returns dicts and the whole
    # downstream pipeline (dedup, filters, threading, Parquet) consumes dicts. A
    # different row type for the second corpus is the first step toward two code
    # paths.
    out: list[dict] = []
    failures: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, mid): mid for mid in message_ids}
        for future in as_completed(futures):
            message_id = futures[future]
            completed += 1
            try:
                parsed = future.result()
                if parsed is not None:
                    out.append(asdict(parsed))
            except GmailError as exc:
                # One unreadable message must not end a 35k-message sync.
                failures.append(f"{message_id}: {exc}")
            if on_progress and completed % 50 == 0:
                on_progress(completed, len(message_ids))
    # One last call so a `total` not divisible by 50 still ends on its real
    # final count rather than leaving a caller's tracked progress stuck a
    # few messages short of "done".
    if on_progress and message_ids:
        on_progress(completed, len(message_ids))
    if failures:
        print(f"  {len(failures)} message(s) could not be fetched; first: "
              f"{failures[0][:120]}")
    return out


def sync(client: GmailClient, state_path: Path, query: str = "",
         max_messages: int = 0, force_full: bool = False, on_progress=None
         ) -> tuple[list[dict], SyncState]:
    """Incremental where possible, full where not.

    Returns the new messages and the updated cursor. The cursor is read *before*
    fetching, so messages arriving during the sync are picked up next time rather
    than being skipped.

    `on_progress`, if given, is forwarded to `fetch_messages` (see its own
    docstring) for the full-sync path; the incremental (`history_since`)
    path calls it directly since it fetches serially rather than through
    `fetch_messages` - same `(completed, total)` shape either way.
    """
    state = SyncState.load(state_path)
    next_cursor = client.current_history_id()

    if state.history_id and not force_full:
        try:
            ids = list(client.history_since(state.history_id))
            print(f"  incremental sync: {len(ids)} new message(s) since "
                  f"historyId {state.history_id}")
            from dataclasses import asdict

            from .enron import parse_message
            ids = ids[:max_messages or None]
            if on_progress:
                on_progress(0, len(ids))
            messages = []
            for i, message_id in enumerate(ids, start=1):
                try:
                    parsed = parse_message(client.raw_message(message_id),
                                           source_path=f"gmail:{message_id}")
                    if parsed is not None:
                        messages.append(asdict(parsed))
                except GmailError:
                    continue
                if on_progress and (i % 50 == 0 or i == len(ids)):
                    on_progress(i, len(ids))
        except HistoryTooOld:
            print("  stored historyId has expired (Google keeps roughly a week) - "
                  "falling back to a full sync")
            messages = fetch_messages(client, query, max_messages, on_progress=on_progress)
    else:
        messages = fetch_messages(client, query, max_messages, on_progress=on_progress)

    state.history_id = next_cursor
    state.last_sync_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state.messages_seen += len(messages)
    return messages, state
