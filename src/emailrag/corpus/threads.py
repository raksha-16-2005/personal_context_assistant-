"""Thread reconstruction.

The obvious approach - follow ``References`` / ``In-Reply-To`` - only gets you
part of the way here. Enron's messages were exported through JavaMail and a
large share carry a ``Message-ID`` but no reply headers at all, so a
references-only threader leaves most of the corpus as singletons and the
thread-aware chunking arm of the ablation would be testing nothing.

So this runs two passes:

1. Union by explicit reply headers, wherever they exist. Highest precision.
2. For everything still unlinked, union messages that share a normalized
   subject *and* at least one participant *and* fall within a time window.
   Subject alone is far too loose - "Re: meeting" recurs for years across
   unrelated groups - and the participant test is what makes it safe.

`stats()` reports how much of the threading came from each pass, which belongs
in EVALUATION.md: a reader should know the second pass is a heuristic.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta

_MSGID = re.compile(r"<[^<>@\s]+@[^<>@\s]+>")

# Messages in the same subject bucket further apart than this are treated as
# unrelated reuse of a common subject line rather than one continuing thread.
WINDOW = timedelta(days=90)

# Pass 2 refuses to thread a subject bucket larger than this.
#
# Without the cap the heuristic chains transitively: a petition where 1,116
# different people mail the same subject to one recipient became a single
# 1,124-message "thread", because each consecutive pair shares the recipient
# and falls inside the window. Measured on Enron, genuine threads have a median
# size of 1 and a p99 of 7, so a 50-message bucket is a broadcast pattern -
# newsletter, automated alert, or mass mailing - not a conversation. Those are
# left as singletons rather than silently fused.
MAX_BUCKET = 50


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _participants(row: dict) -> set[str]:
    out: set[str] = set()
    for field in ("sender", "recipients", "cc"):
        value = row.get(field) or ""
        out.update(a for a in value.split(";") if a)
    return out


def assign_threads(rows: list[dict]) -> tuple[list[str], dict[str, int]]:
    """Return (thread_id per row, stats). Input order is preserved."""
    n = len(rows)
    dsu = _DSU(n)
    by_msgid = {r["message_id"]: i for i, r in enumerate(rows) if r.get("message_id")}

    # Pass 1 - explicit reply headers.
    linked_by_header = 0
    for i, row in enumerate(rows):
        refs = f"{row.get('in_reply_to') or ''} {row.get('references') or ''}"
        for ref in _MSGID.findall(refs):
            j = by_msgid.get(ref)
            if j is not None and dsu.union(i, j):
                linked_by_header += 1

    # Pass 2 - subject + participant overlap + time window, for the remainder.
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        subject = row.get("subject_norm") or ""
        if len(subject) >= 4:  # "fyi", "hi" carry no thread signal
            buckets[subject].append(i)

    linked_by_heuristic = 0
    skipped_buckets = 0
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        if len(idxs) > MAX_BUCKET:
            skipped_buckets += 1
            continue
        dated = sorted(idxs, key=lambda i: (rows[i].get("date_utc") is None,
                                            rows[i].get("date_utc")))
        # Compare each message only against its immediate predecessor in time:
        # O(n) per bucket, and a genuine thread is transitively connected
        # through it anyway.
        for prev, cur in zip(dated, dated[1:]):
            d_prev, d_cur = rows[prev].get("date_utc"), rows[cur].get("date_utc")
            if d_prev is None or d_cur is None or (d_cur - d_prev) > WINDOW:
                continue
            if _participants(rows[prev]) & _participants(rows[cur]):
                if dsu.union(prev, cur):
                    linked_by_heuristic += 1

    roots = [dsu.find(i) for i in range(n)]
    sizes: dict[int, int] = defaultdict(int)
    for r in roots:
        sizes[r] += 1

    # Stable, content-derived thread ids: the message_id of the earliest
    # message in the thread. Reproducible across runs, unlike an ordinal.
    earliest: dict[int, tuple] = {}
    for i, r in enumerate(roots):
        key = (rows[i].get("date_utc") is None, rows[i].get("date_utc"), rows[i]["dedup_key"])
        if r not in earliest or key < earliest[r][0]:
            earliest[r] = (key, rows[i]["dedup_key"])
    thread_ids = [earliest[r][1] for r in roots]

    multi = sum(1 for s in sizes.values() if s > 1)
    stats = {
        "messages": n,
        "threads": len(sizes),
        "multi_message_threads": multi,
        "singletons": len(sizes) - multi,
        "linked_by_header": linked_by_header,
        "linked_by_heuristic": linked_by_heuristic,
        "oversized_buckets_skipped": skipped_buckets,
        "largest_thread": max(sizes.values()) if sizes else 0,
    }
    return thread_ids, stats
