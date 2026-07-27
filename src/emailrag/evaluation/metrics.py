"""IR metrics, scored at message level with binary relevance.

Two decisions worth defending in EVALUATION.md:

*Message level, not chunk level.* The eval set labels which **messages** answer
a query. A retriever is credited once for surfacing a relevant message, no
matter how many of its chunks appear in the ranking. Scoring chunks instead
would let a strategy that emits four chunks per message beat one that emits a
single chunk purely by occupying more of the top-k - an artifact of chunk
granularity, not retrieval quality, and it would invalidate dimension 1.

*Binary relevance.* Graded judgements over 80 queries would need a second
annotator to be credible, and inter-annotator agreement on a graded scale for
a corpus one person hand-labels alone is not defensible. Binary keeps nDCG
honest: gain is 1 for a labelled message and 0 otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def collapse_to_messages(ranked_chunks: list[tuple[str, float]]) -> list[str]:
    """Chunk ranking -> message ranking, keeping each message's best position.

    Input is [(chunk_id, score), ...] in descending score order, where a
    chunk_id is "{dedup_key}:{ordinal}".
    """
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id, _score in ranked_chunks:
        key = chunk_id.rsplit(":", 1)[0]
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")   # undefined, not zero - unanswerable controls
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1)
              for i, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    # Ideal ranking puts min(|rel|, k) relevant docs in the top positions.
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(slots=True)
class QueryResult:
    query_id: str
    query_class: str
    ranked_messages: list[str]
    relevant: set[str]
    latency_ms: float

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def aggregate(results: list[QueryResult], ks: tuple[int, ...] = (5, 20)) -> dict:
    """Mean metrics over answerable queries, plus p50/p95 latency over all.

    Unanswerable controls are excluded from recall/MRR/nDCG - there is no
    correct document to retrieve, so those metrics are undefined rather than
    zero. Including them as zeros would silently drag every score down by
    12.5% and make configs incomparable to any published baseline. They are
    scored separately, on refusal, in the generation eval.
    """
    answerable = [r for r in results if r.answerable]
    out: dict[str, float | int] = {"n_queries": len(results), "n_answerable": len(answerable)}

    if answerable:
        for k in ks:
            out[f"recall@{k}"] = sum(
                recall_at_k(r.ranked_messages, r.relevant, k) for r in answerable
            ) / len(answerable)
        out["mrr"] = sum(
            reciprocal_rank(r.ranked_messages, r.relevant) for r in answerable
        ) / len(answerable)
        out["ndcg@10"] = sum(
            ndcg_at_k(r.ranked_messages, r.relevant, 10) for r in answerable
        ) / len(answerable)

    lat = sorted(r.latency_ms for r in results)
    if lat:
        out["p50_ms"] = lat[len(lat) // 2]
        out["p95_ms"] = lat[min(int(len(lat) * 0.95), len(lat) - 1)]
    return out


def by_query_class(results: list[QueryResult], ks: tuple[int, ...] = (5, 20)) -> dict[str, dict]:
    """Per-class breakdown - this is the table that carries the router result.

    The headline claim (dense retrieval collapses on temporal queries while
    SQL answers them exactly) is only visible when metrics are split this way;
    a single corpus-wide average hides it completely.
    """
    buckets: dict[str, list[QueryResult]] = {}
    for r in results:
        buckets.setdefault(r.query_class, []).append(r)
    return {cls: aggregate(rs, ks) for cls, rs in sorted(buckets.items())}
