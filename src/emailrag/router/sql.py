"""The SQL arm: turning a time window into a query over `commitments`.

This is the half of the router that retrieval cannot do. "What's due next week" is
`WHERE due_at BETWEEN '2001-11-05' AND '2001-11-11'` - an exact answer, complete
and ordered, with no notion of similarity involved. Embedding the same question
returns messages that *talk about* deadlines, which is a different set and is
usually wrong.

**Windows resolve against `as_of`, never against now.** The corpus is frozen in
1999-2002, so "next week" has no meaning without an anchor; the eval set carries
`as_of` per temporal query for exactly this reason, and the same anchor that
resolves a query's window is the one that resolved the commitment's due date at
extraction time. Using `datetime.now()` here would make every temporal answer
empty and the failure would look like a retrieval problem.

**Queries are parameterised.** The window is computed in Python and bound; no part
of a user question is ever interpolated into SQL. That is not paranoia about a
local demo - phase 7 points this at a real mailbox, and a question is untrusted
input from the moment it can come from a web form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..extraction import dates as D


@dataclass(slots=True)
class TemporalQuery:
    """A resolved window plus the filters that came with it."""

    start: date
    end: date
    phrase: str = ""
    include_overdue: bool = False
    unresolved: bool = False       # a window was implied but could not be pinned

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def describe(self) -> str:
        overdue = " plus anything already overdue" if self.include_overdue else ""
        return (f"{self.start.isoformat()} to {self.end.isoformat()} "
                f"({self.days} days){overdue}")


# Window phrases, resolved via the same machinery that resolves commitment due
# dates - so a query's "next week" and a message's "next week" cannot disagree.
_WINDOWS: list[tuple[str, re.Pattern]] = [
    ("this_week", re.compile(r"\bthis week\b", re.I)),
    ("next_week", re.compile(r"\bnext week\b", re.I)),
    ("last_week", re.compile(r"\blast week\b", re.I)),
    ("this_month", re.compile(r"\bthis month\b", re.I)),
    ("next_month", re.compile(r"\bnext month\b", re.I)),
    ("last_month", re.compile(r"\blast month\b", re.I)),
    ("this_quarter", re.compile(r"\bthis quarter\b", re.I)),
    ("today", re.compile(r"\b(?:today|by end of (?:the )?day|eod)\b", re.I)),
    ("yesterday", re.compile(r"\byesterday\b", re.I)),
    ("tomorrow", re.compile(r"\btomorrow\b", re.I)),
    ("next_n_days", re.compile(r"\b(?:next|coming|in the next)\s+(\d+)\s+days?\b", re.I)),
    ("last_n_days", re.compile(r"\b(?:last|past|previous)\s+(\d+)\s+days?\b", re.I)),
]

_OVERDUE = re.compile(r"\b(?:overdue|late|past due|still (?:open|outstanding))\b", re.I)


def _week_bounds(anchor: date, offset_weeks: int = 0) -> tuple[date, date]:
    monday = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=offset_weeks)
    return monday, monday + timedelta(days=6)


def _month_bounds(anchor: date, offset_months: int = 0) -> tuple[date, date]:
    month = anchor.month - 1 + offset_months
    year = anchor.year + month // 12
    first = date(year, month % 12 + 1, 1)
    nxt = date(first.year + (first.month == 12), (first.month % 12) + 1, 1)
    return first, nxt - timedelta(days=1)


def parse_window(query: str, as_of: date | datetime | str) -> TemporalQuery | None:
    """The window a question asks about, anchored on `as_of`.

    Returns None when the question names no window. A temporal question with no
    resolvable window is not the same as one with an empty window, and the caller
    should say so rather than return "nothing is due".
    """
    anchor = _as_date(as_of)
    if anchor is None:
        return None
    include_overdue = bool(_OVERDUE.search(query or ""))

    for name, pattern in _WINDOWS:
        match = pattern.search(query or "")
        if not match:
            continue
        start, end = _bounds(name, match, anchor)
        return TemporalQuery(start=start, end=end, phrase=match.group(0),
                             include_overdue=include_overdue)

    # No window phrase, but an explicit date might still be in there - "what is
    # due by November 8" is a window ending on a date.
    resolved = D.try_resolve(query or "", anchor)
    if resolved is not None:
        return TemporalQuery(start=anchor, end=resolved.value,
                             phrase=resolved.phrase or resolved.rule,
                             include_overdue=include_overdue)

    if include_overdue:
        # "What's overdue" is a window with no upper bound the question states:
        # everything up to the anchor.
        return TemporalQuery(start=date(1900, 1, 1), end=anchor,
                             phrase="overdue", include_overdue=True)
    return None


def _bounds(name: str, match: re.Match, anchor: date) -> tuple[date, date]:
    if name == "this_week":
        return _week_bounds(anchor, 0)
    if name == "next_week":
        return _week_bounds(anchor, 1)
    if name == "last_week":
        return _week_bounds(anchor, -1)
    if name == "this_month":
        return _month_bounds(anchor, 0)
    if name == "next_month":
        return _month_bounds(anchor, 1)
    if name == "last_month":
        return _month_bounds(anchor, -1)
    if name == "this_quarter":
        first_month = ((anchor.month - 1) // 3) * 3 + 1
        start = date(anchor.year, first_month, 1)
        _, end = _month_bounds(start, 2)
        return start, end
    if name == "today":
        return anchor, anchor
    if name == "yesterday":
        d = anchor - timedelta(days=1)
        return d, d
    if name == "tomorrow":
        d = anchor + timedelta(days=1)
        return d, d
    if name == "next_n_days":
        return anchor, anchor + timedelta(days=int(match.group(1)))
    if name == "last_n_days":
        return anchor - timedelta(days=int(match.group(1))), anchor
    raise ValueError(f"unhandled window {name!r}")


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


# The window is bound as parameters, never formatted into the string. A question is
# untrusted input from the moment it can arrive from a web form, which it can as
# soon as phase 7 lands.
WINDOW_SQL = """
SELECT message_id, text, kind, direction, owner, counterparty,
       due_phrase, due_at, due_precision, due_ambiguous, confidence
  FROM commitments
 WHERE model = %(model)s
   AND due_at IS NOT NULL
   AND due_at BETWEEN %(start)s AND %(end)s
 ORDER BY due_at, confidence DESC
 LIMIT %(limit)s;
"""

COUNT_SQL = """
SELECT count(*) AS n, min(due_at) AS earliest, max(due_at) AS latest
  FROM commitments
 WHERE model = %(model)s
   AND due_at IS NOT NULL
   AND due_at BETWEEN %(start)s AND %(end)s;
"""


def window_sql(window: TemporalQuery, model: str, limit: int = 200) -> tuple[str, dict]:
    """The SQL and its parameters for one resolved window."""
    return WINDOW_SQL, {"model": model, "start": window.start,
                        "end": window.end, "limit": limit}


def filter_commitments(commitments: list, window: TemporalQuery) -> list:
    """The same filter, in memory.

    Postgres is optional in this project - `index/store.py` exists but the served
    demo runs off JSONL - and the router has to be measurable without a database
    up. This is the fallback path, and it is deliberately the same predicate as
    WINDOW_SQL so the two cannot drift.
    """
    out = []
    for c in commitments:
        due = getattr(c, "due_at", None)
        if due is None:
            continue
        if isinstance(due, datetime):
            due = due.date()
        if window.start <= due <= window.end:
            out.append(c)
    return sorted(out, key=lambda c: (c.due_at, -getattr(c, "confidence", 0.0)))


def filter_messages_by_date(messages: dict, window: TemporalQuery, tz=None) -> list[dict]:
    """Messages actually *received* inside `window` - not a commitment's due
    date, the message's own `date_utc`.

    A commitment is a fact extracted from one message; "what arrived today"
    is a question about messages that may never have had anything extracted
    from them at all. `filter_commitments` cannot answer it regardless of
    how it's tuned, because it only ever sees the messages extraction found
    an obligation in. `messages` here is `Pipeline._messages` - already
    loaded for every message this index has (see that method's own
    docstring) - so this needs no new index, no new load, and no database.

    `tz` has to be the same zone `window` was resolved in (a user's local
    day in the multi-tenant web app, not necessarily UTC - see
    chat/routes.py). Comparing a UTC calendar date against a window anchored
    on the user's local "today" is wrong for roughly a third of every day -
    whenever the two dates disagree, which is exactly the gap between UTC
    midnight and the user's own midnight. `date_utc` is the full timestamp
    for exactly this reason; converting it into `tz` before taking `.date()`
    is what makes "today" mean the same day on both sides of the comparison.

    Returned newest-first: "what did I get today" reads as a list, not a
    ranking, and the newest arrival is the one most worth seeing first.
    """
    from datetime import timezone as _timezone

    zone = tz or _timezone.utc
    out = []
    for dedup_key, row in messages.items():
        raw = row.get("date_utc")
        if raw is None:
            continue
        received = raw.astimezone(zone).date()
        if window.start <= received <= window.end:
            out.append({**row, "dedup_key": dedup_key})
    return sorted(out, key=lambda r: r["date"], reverse=True)
