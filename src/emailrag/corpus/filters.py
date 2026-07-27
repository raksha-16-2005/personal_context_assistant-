"""Bulk-mail rules filter. Deliberately not ML - see plan section 1.

Calibration note: the plan inherited a "~80% is bulk" figure from modern Gmail.
That does not hold for a 1999-2002 corporate corpus. Enron's bulk share is
single-digit percent; dedup does the real reduction. `report()` prints the
actual rate so the README can state the measured number rather than the
assumed one.
"""
from __future__ import annotations

import re

_NOREPLY = re.compile(
    r"(?:^|[._-])(?:no[._-]?reply|do[._-]?not[._-]?reply|notification|notifications"
    r"|mailer[._-]?daemon|postmaster|bounce|listserv|majordomo|automail|autoreply)",
    re.IGNORECASE,
)

# Enron-era internal broadcast senders. These are announcement blasts to the
# whole company, not correspondence, and they crowd out real threads.
_BULK_SENDERS = frozenset({
    "enron.announcements@enron.com",
    "enron_announcements@enron.com",
    "announcements.enron@enron.com",
    "outlook.team@enron.com",
    "ubsw.energy.announcements@enron.com",
    "perfmgmt@enron.com",
    "ezonline@enron.com",
})

_BULK_SUBJECTS = re.compile(
    r"^(?:\s*(?:re|fw|fwd)\s*:\s*)*\s*(?:"
    r"out of (?:the )?office|automatic reply|undeliverable|delivery status notification"
    r"|returned mail|read:|tw:|newsletter|unsubscribe"
    r")",
    re.IGNORECASE,
)


def is_bulk(msg: dict) -> bool:
    """True if the message is machine-generated or a broadcast blast."""
    if msg.get("has_list_unsubscribe"):
        return True
    sender = (msg.get("sender") or "").lower()
    if sender in _BULK_SENDERS or _NOREPLY.search(sender):
        return True
    if _BULK_SUBJECTS.match(msg.get("subject") or ""):
        return True
    # Empty-bodied messages carry no retrievable content.
    if len((msg.get("body_new") or "").strip()) < 20:
        return True
    return False


def report(total: int, dropped: int) -> str:
    pct = (dropped / total * 100) if total else 0.0
    return f"rules filter: dropped {dropped:,} / {total:,} ({pct:.1f}%)"


# --- corpus-level filter ---------------------------------------------------
#
# `is_bulk` looks at one message and cannot see that the same sender emitted
# the identical subject 623 times. That pattern - automated alerts, newsletters,
# quota warnings - is only visible in aggregate, and it does real damage
# downstream: these blasts dominate their subject bucket and make the threading
# heuristic chain hundreds of unrelated messages into one "thread".
#
# Threshold is on (sender, normalized subject) pairs. A human thread reuses a
# subject a handful of times; 25 identical subjects from one sender is a
# machine. Measured on Enron the cut is unambiguous - genuine threads sit at a
# median of 1 and p99 of 7.

RECURRING_THRESHOLD = 25


def find_recurring_blasts(pairs: list[tuple[str, str]], threshold: int = RECURRING_THRESHOLD
                          ) -> set[tuple[str, str]]:
    """Return the (sender, subject_norm) pairs that recur above `threshold`."""
    counts: dict[tuple[str, str], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    return {pair for pair, n in counts.items() if n >= threshold and pair[1]}
