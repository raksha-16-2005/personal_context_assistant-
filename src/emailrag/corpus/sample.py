"""Stratified subsample used for every ablation run.

Ordering matters here. The subsample is drawn **before** the eval set is
written, and the eval queries are then authored against the subsample. Doing
it the other way - sample a corpus, then discover the gold messages for some
queries are outside it - either silently caps recall at less than 1.0 or
forces the sample to be patched around the labels, which biases it.

Two invariants:

**Whole threads, never partial.** A thread split across the sample boundary
would break thread-aware chunking and make multi-hop queries unanswerable for
reasons unrelated to retrieval quality.

**Mirror the corpus, don't flatten it.** Sampling threads uniformly would
over-represent singletons, which are ~most of the thread count but a minority
of messages. Strata are (year x thread-size bucket) with per-stratum quotas
proportional to that stratum's share of *messages* in the full corpus, so the
sample's message-level composition matches the source.
"""
from __future__ import annotations

import random
from collections import defaultdict

SIZE_BUCKETS = ((1, "singleton"), (5, "small"), (20, "medium"))


def _size_bucket(n: int) -> str:
    for upper, name in SIZE_BUCKETS:
        if n <= upper:
            return name
    return "large"


def _stratum(rows: list[dict]) -> str:
    years = [r["date_utc"].year for r in rows if r.get("date_utc") is not None]
    # Enron's usable span is 1999-2002; anything outside is a bad Date header.
    year = min(max(min(years), 1998), 2003) if years else 0
    return f"{year}:{_size_bucket(len(rows))}"


def stratified_thread_sample(
    messages: list[dict],
    target_messages: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Return (sampled messages, stats). Threads are kept intact."""
    threads: dict[str, list[dict]] = defaultdict(list)
    for m in messages:
        threads[m["thread_id"]].append(m)

    strata: dict[str, list[str]] = defaultdict(list)
    for tid, rows in threads.items():
        strata[_stratum(rows)].append(tid)

    total = len(messages)
    if target_messages >= total:
        return list(messages), {"sampled_messages": total, "sampled_threads": len(threads),
                                "note": "target >= corpus; returned everything"}

    rng = random.Random(seed)
    chosen: list[str] = []
    n_selected = 0

    # Quota per stratum is proportional to its share of messages, not threads.
    for name in sorted(strata):
        tids = sorted(strata[name])
        rng.shuffle(tids)
        stratum_msgs = sum(len(threads[t]) for t in tids)
        quota = target_messages * stratum_msgs / total
        taken = 0
        for tid in tids:
            if taken >= quota:
                break
            chosen.append(tid)
            taken += len(threads[tid])
            n_selected += len(threads[tid])

    # Proportional quotas round down; top up deterministically to close the gap.
    if n_selected < target_messages:
        remaining = sorted(set(threads) - set(chosen))
        rng.shuffle(remaining)
        for tid in remaining:
            if n_selected >= target_messages:
                break
            chosen.append(tid)
            n_selected += len(threads[tid])

    chosen_set = set(chosen)
    sampled = [m for m in messages if m["thread_id"] in chosen_set]
    stats = {
        "corpus_messages": total,
        "corpus_threads": len(threads),
        "sampled_messages": len(sampled),
        "sampled_threads": len(chosen_set),
        "strata": len(strata),
    }
    return sampled, stats
