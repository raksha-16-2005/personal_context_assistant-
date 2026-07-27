"""Hybrid fusion - the two arms of dimension 3.

**RRF** ignores scores and uses only ranks, which is exactly why it is the
robust default: BM25 returns unbounded positive scores and cosine similarity
returns [-1, 1], and there is no principled way to compare them directly.

**Weighted score fusion** does compare them, after normalization, and can beat
RRF when one retriever is reliably stronger - but it needs a normalization
choice and a weight, both of which are extra tuning surface. Including both
is the point of the ablation: if weighted fusion does not beat RRF by enough
to justify tuning a weight per corpus, that is a result worth publishing.

Min-max normalization is applied per result list, not globally. A global
normalizer would need corpus-wide score bounds, which change with every index
rebuild and would make runs incomparable.
"""
from __future__ import annotations

RRF_K = 60  # Cormack et al. 2009; deliberately left at the published default


def reciprocal_rank_fusion(
    runs: list[list[tuple[str, float]]],
    k: int = RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked runs by rank alone. Scores in the input are ignored."""
    scores: dict[str, float] = {}
    for run in runs:
        for rank, (doc_id, _score) in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Tie-break on doc_id so equal-score docs order deterministically.
    out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:top_k] if top_k else out


def _min_max(run: list[tuple[str, float]]) -> dict[str, float]:
    if not run:
        return {}
    values = [s for _, s in run]
    lo, hi = min(values), max(values)
    if hi == lo:
        # Degenerate list (single hit, or all-equal scores): treat every doc as
        # maximally relevant for this retriever rather than dividing by zero.
        return {d: 1.0 for d, _ in run}
    return {d: (s - lo) / (hi - lo) for d, s in run}


def weighted_score_fusion(
    runs: list[list[tuple[str, float]]],
    weights: list[float],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalize each run, then take a weighted sum.

    A document missing from a run contributes 0 for that retriever, which is
    the correct reading: it fell outside that retriever's candidate set.
    """
    if len(runs) != len(weights):
        raise ValueError(f"got {len(runs)} runs but {len(weights)} weights")

    scores: dict[str, float] = {}
    for run, weight in zip(runs, weights):
        for doc_id, norm in _min_max(run).items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * norm

    out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:top_k] if top_k else out
