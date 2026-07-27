"""Resolving "next Thursday" to an absolute date.

This is the single largest source of silent wrongness in commitment extraction,
and it gets its own accuracy metric because of it. A due date that is wrong by
seven days looks exactly as confident as one that is right.

**The model finds the phrase; Python does the arithmetic.** Language models are
unreliable at date arithmetic - they will confidently return a Thursday that is
not a Thursday, or add seven days to the wrong week - and their errors are
invisible without recomputing the answer anyway. So the extractor's job is to
identify the deadline *phrase* and classify it; resolution happens here, in code
that can be unit-tested against a calendar. This also means the date-accuracy
metric measures language understanding rather than the model's ability to count.

**Every resolution is relative to the message's own sent date**, never to now. A
1999-2002 corpus has no "now", and a commitment extracted today from a mail sent
in 2001 is due relative to 2001.

**"Next Thursday" is genuinely ambiguous and is reported as such.** Sent on a
Tuesday, it means the Thursday two days away to some writers and the Thursday
nine days away to others; US business usage does not converge. The convention
here is the following calendar week, because "this Thursday" is the idiom for the
nearer one - but the resolution is flagged `ambiguous` and carries the
alternative. A metric that scored these as simply right or wrong would be
measuring the annotator's convention, not the extractor.

Timezone: a deadline stated as a weekday is a *date*, not an instant. Resolutions
carry a `date` plus `precision`, and `resolved_utc` is midnight UTC of that date -
a lower bound, not a claim about the hour. Anything that needs "end of business"
should apply that policy itself rather than have it baked in here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# How far back an explicit month/day with no year may resolve before it is read as
# next year instead. A mail sent 30 December saying "by January 3" means the
# coming January; the same mail saying "the January 3 filing" might mean the past
# one, and that case is genuinely undecidable from the phrase alone. 45 days is
# the threshold, and the flip is recorded as `rolled_year` so the metric can
# report how often it fired.
BACKWARD_TOLERANCE_DAYS = 45

# Precision of a resolved date. A deadline of "sometime next month" is not a day,
# and treating it as one manufactures precision the message never had.
PRECISION_DAY = "day"
PRECISION_WEEK = "week"
PRECISION_MONTH = "month"
PRECISION_QUARTER = "quarter"


@dataclass(slots=True)
class Resolved:
    """One resolved deadline."""

    value: date
    precision: str = PRECISION_DAY
    phrase: str = ""
    rule: str = ""
    ambiguous: bool = False
    alternative: date | None = None
    rolled_year: bool = False

    @property
    def resolved_utc(self) -> datetime:
        """Midnight UTC of the resolved date - a lower bound on the deadline,
        not a claim about the hour."""
        return datetime.combine(self.value, time.min, tzinfo=timezone.utc)

    @property
    def iso(self) -> str:
        return self.value.isoformat()


class UnresolvableDate(ValueError):
    """The phrase carries no recoverable date."""


def _as_date(sent: date | datetime) -> date:
    return sent.date() if isinstance(sent, datetime) else sent


def _next_weekday(sent: date, target: int, *, allow_today: bool = False) -> date:
    """The next `target` weekday strictly after `sent` (or on it, if allowed)."""
    delta = (target - sent.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return sent + timedelta(days=delta)


def _following_week_weekday(sent: date, target: int) -> date:
    """That weekday in the calendar week after the sent date's week."""
    monday_next = sent + timedelta(days=7 - sent.weekday())
    return monday_next + timedelta(days=target)


def _end_of_month(d: date) -> date:
    first_next = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
    return first_next - timedelta(days=1)


def _end_of_quarter(d: date) -> date:
    last_month = ((d.month - 1) // 3) * 3 + 3
    return _end_of_month(date(d.year, last_month, 1))


# Patterns are tried in order, and the order is load-bearing. Every one of these
# is a longer phrase that contains a shorter rule's trigger:
#
#   "day after tomorrow"  contains  "tomorrow"
#   "end of next week"    contains  "next week"
#   "next Thursday"       contains  a bare weekday
#
# Testing the short rule first silently returns a date that is off by one day, one
# week, or seven days - which is exactly the class of error this module exists to
# prevent. Longest and most specific first.
_RULES: list[tuple[str, re.Pattern]] = [
    ("day_after_tomorrow", re.compile(r"\bday after tomorrow\b")),
    ("today", re.compile(r"\b(today|end of (?:the )?day|eod|cob|by close of business)\b")),
    ("tomorrow", re.compile(r"\b(tomorrow|tmrw)\b")),
    ("yesterday", re.compile(r"\byesterday\b")),
    ("end_of_next_week", re.compile(r"\bend of next week\b")),
    ("end_of_week", re.compile(r"\bend of (?:the |this )?week\b|\beow\b")),
    ("end_of_next_month", re.compile(r"\bend of next month\b")),
    ("end_of_month", re.compile(r"\bend of (?:the |this )?month\b|\beom\b")),
    ("end_of_quarter", re.compile(r"\bend of (?:the |this )?quarter\b|\beoq\b")),
    ("next_weekday", re.compile(r"\bnext\s+(" + "|".join(WEEKDAYS) + r")\b")),
    ("this_weekday", re.compile(r"\bthis\s+(" + "|".join(WEEKDAYS) + r")\b")),
    # "within N days" is the same deadline as "in N days" - both name the last
    # acceptable date - and it is at least as common in this corpus.
    ("in_n_units", re.compile(
        r"\b(?:in|within)\s+(\d+|a|one|two|three|four|five|six|seven)\s+"
        r"(business day|business days|working day|working days|"
        r"day|days|week|weeks|month|months)\b")),
    ("n_units_from_now", re.compile(
        r"\b(\d+|a|one|two|three|four|five|six|seven)\s+"
        r"(day|days|week|weeks|month|months)\s+from\s+(?:now|today)\b")),
    ("next_week", re.compile(r"\bnext week\b")),
    ("this_week", re.compile(r"\bthis week\b")),
    ("next_month", re.compile(r"\bnext month\b")),
    ("this_month", re.compile(r"\bthis month\b")),
    ("next_quarter", re.compile(r"\bnext quarter\b")),
    ("month_day", re.compile(
        r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,?\s*(\d{4}))?\b")),
    ("day_month", re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(MONTHS) + r")\.?"
        r"(?:,?\s*(\d{4}))?\b")),
    ("iso_date", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("numeric_date", re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")),
    ("bare_weekday", re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b")),
]

_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7}


def _add_business_days(start: date, n: int) -> date:
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def resolve(phrase: str, sent: date | datetime) -> Resolved:
    """Resolve a deadline phrase against the date the message was sent.

    Raises `UnresolvableDate` when nothing in the phrase names a time. Callers
    should treat that as "no due date", not as an error worth retrying: plenty of
    real commitments have no deadline at all.
    """
    if not phrase or not phrase.strip():
        raise UnresolvableDate("empty phrase")
    sent_date = _as_date(sent)
    text = phrase.lower().strip()

    for rule, pattern in _RULES:
        match = pattern.search(text)
        if match:
            return _apply(rule, match, sent_date, phrase)

    raise UnresolvableDate(f"no date expression in {phrase!r}")


def _apply(rule: str, match: re.Match, sent: date, phrase: str) -> Resolved:
    mk = lambda value, **kw: Resolved(value=value, phrase=phrase, rule=rule, **kw)  # noqa: E731

    if rule == "today":
        return mk(sent)
    if rule == "tomorrow":
        return mk(sent + timedelta(days=1))
    if rule == "day_after_tomorrow":
        return mk(sent + timedelta(days=2))
    if rule == "yesterday":
        return mk(sent - timedelta(days=1))

    if rule == "end_of_week":
        return mk(_next_weekday(sent, 4, allow_today=True))
    if rule == "end_of_next_week":
        return mk(_following_week_weekday(sent, 4))
    if rule == "end_of_month":
        return mk(_end_of_month(sent))
    if rule == "end_of_next_month":
        nxt = _end_of_month(sent) + timedelta(days=1)
        return mk(_end_of_month(nxt))
    if rule == "end_of_quarter":
        return mk(_end_of_quarter(sent))

    if rule == "next_weekday":
        target = WEEKDAYS[match.group(1)]
        # The convention: the following calendar week. Flagged ambiguous with the
        # nearer reading attached - see the module docstring.
        chosen = _following_week_weekday(sent, target)
        nearer = _next_weekday(sent, target)
        return mk(chosen, ambiguous=chosen != nearer, alternative=nearer)

    if rule in ("this_weekday", "bare_weekday"):
        return mk(_next_weekday(sent, WEEKDAYS[match.group(1)]))

    if rule in ("in_n_units", "n_units_from_now"):
        raw = match.group(1)
        n = int(raw) if raw.isdigit() else _WORD_NUMBERS[raw]
        unit = match.group(2)
        if unit.startswith(("business", "working")):
            return mk(_add_business_days(sent, n))
        if unit.startswith("day"):
            return mk(sent + timedelta(days=n))
        if unit.startswith("week"):
            return mk(sent + timedelta(weeks=n))
        # Months: calendar months, clamped to the end of a short target month so
        # "in one month" from 31 January is 28/29 February rather than 3 March.
        month = sent.month - 1 + n
        year = sent.year + month // 12
        month = month % 12 + 1
        day = min(sent.day, _end_of_month(date(year, month, 1)).day)
        return mk(date(year, month, day), precision=PRECISION_MONTH)

    if rule == "next_week":
        return mk(_following_week_weekday(sent, 4), precision=PRECISION_WEEK)
    if rule == "this_week":
        return mk(_next_weekday(sent, 4, allow_today=True), precision=PRECISION_WEEK)
    if rule == "next_month":
        nxt = _end_of_month(sent) + timedelta(days=1)
        return mk(_end_of_month(nxt), precision=PRECISION_MONTH)
    if rule == "this_month":
        return mk(_end_of_month(sent), precision=PRECISION_MONTH)
    if rule == "next_quarter":
        after = _end_of_quarter(sent) + timedelta(days=1)
        return mk(_end_of_quarter(after), precision=PRECISION_QUARTER)

    if rule == "iso_date":
        y, m, d = (int(match.group(i)) for i in (1, 2, 3))
        return mk(date(y, m, d))

    if rule in ("month_day", "day_month"):
        if rule == "month_day":
            month, day, year = MONTHS[match.group(1)], int(match.group(2)), match.group(3)
        else:
            day, month, year = int(match.group(1)), MONTHS[match.group(2)], match.group(3)
        return _with_year(month, day, year, sent, phrase, rule)

    if rule == "numeric_date":
        # US convention (month/day), which is what this corpus is. A day > 12 in
        # the first position would contradict it; that is unresolvable rather than
        # silently reinterpreted, because guessing produces a plausible wrong date.
        month, day, year = int(match.group(1)), int(match.group(2)), match.group(3)
        if month > 12:
            raise UnresolvableDate(
                f"{match.group(0)!r} is not month/day; day/month ordering cannot "
                f"be assumed for this corpus")
        return _with_year(month, day, year, sent, phrase, rule)

    raise UnresolvableDate(f"unhandled rule {rule!r}")


def _with_year(month: int, day: int, year: str | None, sent: date, phrase: str,
               rule: str) -> Resolved:
    if year:
        y = int(year)
        if y < 100:                       # two-digit years in a 1999-2002 corpus
            y += 2000 if y < 70 else 1900
        return Resolved(value=_safe_date(y, month, day), phrase=phrase, rule=rule)

    candidate = _safe_date(sent.year, month, day)
    rolled = False
    if (sent - candidate).days > BACKWARD_TOLERANCE_DAYS:
        candidate = _safe_date(sent.year + 1, month, day)
        rolled = True
    return Resolved(value=candidate, phrase=phrase, rule=rule, rolled_year=rolled)


def _safe_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise UnresolvableDate(f"{year}-{month:02d}-{day:02d} is not a real date") from exc


def try_resolve(phrase: str, sent: date | datetime) -> Resolved | None:
    """`resolve` that returns None instead of raising - most messages have no
    deadline, and that is not an exceptional condition."""
    try:
        return resolve(phrase, sent)
    except UnresolvableDate:
        return None


# -- the pre-filter ---------------------------------------------------------

# Cheap gate before the extractor sees a message. Ollama is CPU-only here at
# ~25 s/message, so a 35k-message mailbox is ~240 hours; most messages contain no
# commitment at all. Same reasoning as the bulk filter, and it is the difference
# between an overnight run and a week.
#
# Tuned for recall, not precision: a false positive costs 25 seconds, a false
# negative loses a commitment permanently and invisibly. Anything date-shaped or
# obligation-shaped passes.
_PREFILTER = re.compile(
    r"\b(?:"
    r"by (?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|next|end of|the)"
    r"|due|deadline|deliver|delivery|submit|send (?:me|us|it|over|by)"
    r"|need(?:s|ed)? (?:to|by|it)|please (?:review|send|confirm|sign|approve|get)"
    r"|action item|follow up|followup|asap|urgent"
    r"|before (?:the )?(?:meeting|call|close|monday|tuesday|wednesday|thursday|friday)"
    r"|no later than|not later than|end of (?:day|week|month|quarter|business)"
    r"|eod|eow|eom|eoq|cob"
    r"|(?:this|next) (?:week|month|quarter|monday|tuesday|wednesday|thursday|friday)"
    r"|tomorrow|yesterday"
    r"|\d{1,2}/\d{1,2}|\d{4}-\d{2}-\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"|i(?:'ll| will) (?:send|get|have|call|review|follow)"
    r"|(?:can|could|would) you (?:send|get|review|confirm|sign|have)"
    r")\b",
    re.IGNORECASE)


def looks_like_commitment(text: str) -> bool:
    """Whether a message is worth an extraction call at all.

    Deliberately generous. Report the pass rate rather than tuning this to look
    precise: the number that matters is how many real commitments it drops, and
    that can only be measured against hand labels.
    """
    return bool(_PREFILTER.search(text or ""))
