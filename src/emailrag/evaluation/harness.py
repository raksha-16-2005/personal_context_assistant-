"""Runs an eval set against a built index and produces the ablation rows.

One design point governs everything here: **retrieve deep in chunks, score
shallow in messages.** Metrics are message-level, but retrieval returns chunks,
and several chunks routinely collapse to the same message. Pulling the top 20
chunks can therefore yield well under 20 distinct messages, which would make
recall@20 silently measure chunk clustering instead of retrieval quality - and
it would penalise exactly the chunking strategies that emit more chunks per
message, corrupting dimension 1. So retrieval depth is set in chunks, several
times deeper than the metric cutoff, and `collapse_to_messages` reduces
afterwards.

Latency is split into three numbers rather than one, because the three stages a
row can contain are not comparable and must not be added silently:

    retrieval_ms   encoding the query, searching, fusing. The only number that
                   is comparable across every dimension.
    rerank_ms      the cross-encoder pass. This is what dimension 4's nDCG gain
                   has to justify, so it is reported beside the gain.
    transform_ms   the LLM round-trip for dimension 5, recorded only when the
                   call actually went to the network. A cached call is ~0 ms,
                   and folding that into a latency column would make the number
                   a function of cache state rather than of the system.

Model load time is excluded throughout: it is paid once per process, not per
query.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..index import embed as E
from ..index.dense import DenseIndex
from ..index.fusion import reciprocal_rank_fusion, weighted_score_fusion
from ..index.sparse import SparseIndex
from .evalset import EvalQuery
from .failures import chunk_ranks_from_run
from .metrics import QueryResult, aggregate, by_query_class, collapse_to_messages

# Chunks pulled per retriever before collapsing. 200 keeps >= 20 distinct
# messages even for the chunkiest strategy on the worst query we have seen.
RETRIEVE_DEPTH = 200

RETRIEVERS = ("bm25", "dense", "hybrid_rrf", "hybrid_weighted")

# Fixed, untuned. Sweeping this on 80 queries and reporting the best would be
# fitting the weight to the test set; `sweep_weights` exists for a footnote,
# not for the headline row.
DEFAULT_DENSE_WEIGHT = 0.5


@dataclass(slots=True)
class RunConfig:
    chunking: str
    model: str
    retriever: str
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    # Dimensions 4 and 5. Names, not objects, so a row is JSON-serialisable and
    # a results file says which arms produced it.
    rerank: str = "none"
    transform: str = "none"

    @property
    def label(self) -> str:
        parts = [self.chunking, self.model.split("/")[-1], self.retriever]
        if self.transform != "none":
            parts.append(f"+{self.transform}")
        if self.rerank != "none":
            parts.append(f"+rerank:{self.rerank}")
        return " | ".join(parts)


def _rank(cfg: RunConfig, query: str, qvec: np.ndarray,
          bm25: SparseIndex, dense: DenseIndex) -> list[tuple[str, float]]:
    if cfg.retriever == "bm25":
        return bm25.search(query, top_k=RETRIEVE_DEPTH)
    if cfg.retriever == "dense":
        return dense.search(qvec, top_k=RETRIEVE_DEPTH)

    sparse_run = bm25.search(query, top_k=RETRIEVE_DEPTH)
    dense_run = dense.search(qvec, top_k=RETRIEVE_DEPTH)
    if cfg.retriever == "hybrid_rrf":
        return reciprocal_rank_fusion([dense_run, sparse_run], top_k=RETRIEVE_DEPTH)
    if cfg.retriever == "hybrid_weighted":
        return weighted_score_fusion(
            [dense_run, sparse_run],
            [cfg.dense_weight, 1.0 - cfg.dense_weight],
            top_k=RETRIEVE_DEPTH,
        )
    raise ValueError(f"unknown retriever {cfg.retriever!r}")


def _rank_transformed(cfg: RunConfig, tq, bm25: SparseIndex, dense: DenseIndex,
                      model, instruction: str) -> list[tuple[str, float]]:
    """Retrieve for a possibly-multi-text transformed query.

    Fusion happens in two stages, and the order matters. The several texts a
    transform produces are fused *within* a modality first (by RRF, over ranks),
    and only then are the dense and sparse sides combined by whatever dimension
    3 says. Fusing all runs in one flat pass would let a 4-rewrite expansion
    outvote the sparse retriever 4:1 in a "50/50 hybrid", so a dimension-5 result
    would silently be a dimension-3 change as well.
    """
    if cfg.retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {cfg.retriever!r}")

    dense_run: list[tuple[str, float]] = []
    sparse_run: list[tuple[str, float]] = []

    if cfg.retriever != "bm25":
        qvecs = E.encode_queries(model, tq.dense_texts, instruction or None)
        runs = [dense.search(v, top_k=RETRIEVE_DEPTH) for v in qvecs]
        dense_run = (runs[0] if len(runs) == 1
                     else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))

    if cfg.retriever != "dense":
        runs = [bm25.search(t, top_k=RETRIEVE_DEPTH) for t in tq.sparse_texts]
        sparse_run = (runs[0] if len(runs) == 1
                      else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))

    if cfg.retriever == "bm25":
        return sparse_run
    if cfg.retriever == "dense":
        return dense_run
    if cfg.retriever == "hybrid_rrf":
        return reciprocal_rank_fusion([dense_run, sparse_run], top_k=RETRIEVE_DEPTH)
    return weighted_score_fusion(
        [dense_run, sparse_run],
        [cfg.dense_weight, 1.0 - cfg.dense_weight],
        top_k=RETRIEVE_DEPTH,
    )


def run(cfg: RunConfig, queries: list[EvalQuery], bm25: SparseIndex,
        dense: DenseIndex, model, instruction: str = "", *,
        transformer=None, reranker=None, rerank_top_k: int = 0,
        texts=None) -> dict:
    """Execute one row of the ablation and return its metrics.

    `transformer` is a `query.transform.QueryTransformer` (dimension 5) and
    `reranker` a `index.rerank.CrossEncoderReranker` (dimension 4). Both are
    optional; with neither, this is the plain retrieval row.

    Reranking runs as a second pass over the stored rankings rather than inline,
    because the chunk *texts* it needs are rebuilt from the corpus (see
    index/chunktext.py) and rebuilding them once for the union of every query's
    top-k costs one parquet read instead of eighty.
    """
    _warm_up(bm25, dense, model, instruction)

    results: list[QueryResult] = []
    rankings: list[list[tuple[str, float]]] = []
    timings: list[dict] = []
    transform_log: list[dict] = []

    for q in queries:
        tq, transform_ms = _apply_transform(transformer, q)

        t0 = time.perf_counter()
        if transformer is None or tq.kind == "none":
            # The single-text path is kept as-is: encoding inside the timer so
            # dense/hybrid latency includes the query-encoding cost a real
            # deployment pays, and no encode at all for BM25, which never
            # consumes one.
            if cfg.retriever == "bm25":
                ranked = bm25.search(q.query, top_k=RETRIEVE_DEPTH)
            else:
                qvec = E.encode_queries(model, [q.query], instruction or None)[0]
                ranked = _rank(cfg, q.query, qvec, bm25, dense)
        else:
            ranked = _rank_transformed(cfg, tq, bm25, dense, model, instruction)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        rankings.append(ranked)
        timings.append({"retrieval_ms": retrieval_ms, "rerank_ms": 0.0,
                        "transform_ms": transform_ms, "n_runs": tq.n_runs})
        if tq.kind != "none":
            transform_log.append({"query_id": q.query_id, "kind": tq.kind,
                                  "degraded": tq.degraded, "n_runs": tq.n_runs})

    if reranker is not None:
        top_k = rerank_top_k or 20
        if texts is None:
            raise ValueError(
                "reranking needs chunk texts; pass texts=ChunkTexts(...) built "
                "with index.chunktext.rebuild for the union of the top-k ids")
        for i, ranked in enumerate(rankings):
            # Scored against the *user's* query, never the transform's output. A
            # HyDE document is a retrieval device - a fabricated email - and
            # scoring passages for similarity to a fabrication would measure
            # agreement with the generator, not relevance to the question.
            rankings[i], rerank_ms = reranker.rerank_timed(
                queries[i].query, ranked, texts, top_k)
            timings[i]["rerank_ms"] = rerank_ms

    for q, ranked, t in zip(queries, rankings, timings):
        results.append(QueryResult(
            query_id=q.query_id,
            query_class=q.query_class,
            ranked_messages=collapse_to_messages(ranked),
            relevant=set(q.relevant_message_ids),
            # The scored latency is what the served path costs per query:
            # retrieval plus reranking. The LLM round-trip is reported
            # separately - see the module docstring.
            latency_ms=t["retrieval_ms"] + t["rerank_ms"],
        ))

    overall = aggregate(results)
    overall.update(_stage_latency(timings))

    return {
        "config": {
            "chunking": cfg.chunking,
            "model": cfg.model,
            "retriever": cfg.retriever,
            "dense_weight": cfg.dense_weight if cfg.retriever == "hybrid_weighted" else None,
            "retrieve_depth": RETRIEVE_DEPTH,
            "rerank": cfg.rerank,
            "rerank_top_k": rerank_top_k if reranker is not None else None,
            "transform": cfg.transform,
        },
        "overall": overall,
        "by_class": by_query_class(results),
        "per_query": [
            {"query_id": r.query_id, "query_class": r.query_class,
             "n_relevant": len(r.relevant),
             "found_at": next((i + 1 for i, m in enumerate(r.ranked_messages)
                               if m in r.relevant), None),
             "latency_ms": round(r.latency_ms, 2),
             "rerank_ms": round(t["rerank_ms"], 2),
             # Chunk-level positions of the labelled messages' chunks. This is
             # the only place the chunk-boundary signal survives - after the
             # collapse to messages it is gone - and dimension 6 cannot tell a
             # chunking failure from a total miss without it. Truncated: the
             # first few are what the taxonomy reads, and a query whose message
             # has 40 chunks would otherwise dominate the results file.
             "relevant_chunk_ranks": chunk_ranks_from_run(ranked, r.relevant)[:8]}
            for r, t, ranked in zip(results, timings, rankings)
        ],
        "transform": _transform_summary(transformer, transform_log),
    }


def _warm_up(bm25: SparseIndex, dense: DenseIndex, model, instruction: str) -> None:
    """Pay every lazy first-call cost before the clock starts.

    A sentence-transformer's first `encode` allocates buffers and materialises
    lazily-loaded weights; measured here as ~1.3 s against ~15 ms warm. Whichever
    row happens to run first in a process would otherwise absorb that and every
    later row would look 40x faster - which is exactly how a baseline row ends up
    looking slower than the transform it is supposed to be compared against.
    Excluding it is the same policy the module docstring states for model load:
    it is paid once per process, not per query.
    """
    try:
        E.encode_queries(model, ["warm up the encoder"], instruction or None)
        bm25.search("warm up the retriever", top_k=1)
    except Exception:                                        # noqa: BLE001
        # A fake model in a test need not support this, and a warm-up failure
        # must never be the thing that fails a benchmark.
        pass


def _apply_transform(transformer, q: EvalQuery):
    """Run the dimension-5 transform, timing only calls that hit the network.

    A cached transform takes microseconds. Reporting that as the transform's
    latency would say the LLM round-trip is free, so only live calls are timed
    and cached ones report 0.0 - which the summary distinguishes by counting
    them separately.
    """
    from ..query.transform import identity

    if transformer is None:
        return identity(q.query), 0.0

    t0 = time.perf_counter()
    tq = transformer(q.query, as_of=q.as_of)
    elapsed = (time.perf_counter() - t0) * 1000

    cached = getattr(getattr(transformer, "_llm", None), "last_cached", False)
    return tq, (0.0 if cached else elapsed)


def _stage_latency(timings: list[dict]) -> dict:
    """p50/p95 per stage. Stages are reported side by side, never summed into
    one opaque number, because only `retrieval` is comparable across rows."""
    out: dict[str, float] = {}
    for stage in ("retrieval_ms", "rerank_ms", "transform_ms"):
        values = sorted(t[stage] for t in timings)
        if not values or not any(values):
            continue
        out[f"p50_{stage}"] = values[len(values) // 2]
        out[f"p95_{stage}"] = values[min(int(len(values) * 0.95), len(values) - 1)]
    return out


def _transform_summary(transformer, log: list[dict]) -> dict | None:
    if transformer is None:
        return None
    stats = transformer.stats
    return {
        "kind": transformer.kind,
        "queries": stats.queries,
        "fired": stats.fired,
        "degraded": stats.degraded,
        "llm_calls": stats.calls,
        "per_query": log,
    }


def sweep_weights(cfg: RunConfig, queries: list[EvalQuery], bm25: SparseIndex,
                  dense: DenseIndex, model, instruction: str = "",
                  weights: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)) -> list[dict]:
    """Weighted-fusion sweep.

    Reported as a footnote only. The best weight here is selected on the same
    queries it is evaluated on, so it is an optimistic upper bound rather than
    a result - with 80 queries a dev/test split would be too noisy to do this
    honestly.
    """
    out = []
    for w in weights:
        row = run(RunConfig(cfg.chunking, cfg.model, "hybrid_weighted", w),
                  queries, bm25, dense, model, instruction)
        out.append(row)
    return out


# --- table rendering -------------------------------------------------------

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict], group_by: str = "retriever") -> str:
    """Render rows as a README table, one line per config."""
    header = f"| {group_by.replace('_', ' ').title()} | R@5 | R@20 | MRR | nDCG@10 | p50 ms | p95 ms |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        o = row["overall"]
        lines.append(
            f"| {row['config'][group_by]} "
            f"| {_fmt(o.get('recall@5'))} | {_fmt(o.get('recall@20'))} "
            f"| {_fmt(o.get('mrr'))} | {_fmt(o.get('ndcg@10'))} "
            f"| {_fmt(o.get('p50_ms'), 0)} | {_fmt(o.get('p95_ms'), 0)} |"
        )
    return "\n".join(lines)


def _delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "—"
    return f"{value - baseline:+.3f}"


def rerank_table(rows: list[dict]) -> str:
    """Dimension 4: nDCG gain *against* the latency it costs.

    The gain column is the whole point. A reranker that adds 0.004 nDCG for
    600 ms is a worse system than no reranker, and a table of absolute nDCG
    values lets that read as an improvement. The baseline is the `none` arm; if
    it was not run, gains are omitted rather than computed against the best row,
    which would make the winner look free.
    """
    base = next((r for r in rows if r["config"].get("rerank") == "none"), None)
    b = base["overall"] if base else {}

    lines = ["| Rerank arm | R@5 | R@20 | nDCG@10 | ΔnDCG | rerank p50 ms | total p95 ms |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        o = row["overall"]
        spec = row["config"].get("rerank_spec") or {}
        arm = row["config"].get("rerank", "none")
        if spec:
            arm = f"{arm} ({spec['max_length']} tok, {spec.get('backend', '')})"
        lines.append(
            f"| {arm} | {_fmt(o.get('recall@5'))} | {_fmt(o.get('recall@20'))} "
            f"| {_fmt(o.get('ndcg@10'))} "
            f"| {_delta(o.get('ndcg@10'), b.get('ndcg@10')) if base else '—'} "
            f"| {_fmt(o.get('p50_rerank_ms'), 0)} | {_fmt(o.get('p95_ms'), 0)} |")
    return "\n".join(lines)


def transform_table(rows: list[dict]) -> str:
    """Dimension 5: quality change, how often the transform fired, and its cost.

    `fired` and `degraded` are columns rather than a footnote. A transform that
    silently fell back to the original query on a third of the eval set is
    reporting the baseline's numbers under its own name, and no amount of nDCG
    makes that interpretable without the count.
    """
    base = next((r for r in rows if r["config"].get("transform") == "none"), None)
    b = base["overall"] if base else {}

    lines = ["| Transform | fired | degraded | runs/query | R@5 | R@20 | nDCG@10 "
             "| ΔnDCG | retrieval p50 ms | llm calls |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        o, t = row["overall"], row.get("transform")
        n_runs = max((q.get("n_runs", 1) for q in (t or {}).get("per_query", [])),
                     default=1)
        fired = f"{t['fired']}/{t['queries']}" if t else "—"
        degraded = str(t["degraded"]) if t else "—"
        lines.append(
            f"| {row['config'].get('transform', 'none')} "
            f"| {fired} | {degraded} | {n_runs} "
            f"| {_fmt(o.get('recall@5'))} | {_fmt(o.get('recall@20'))} "
            f"| {_fmt(o.get('ndcg@10'))} "
            f"| {_delta(o.get('ndcg@10'), b.get('ndcg@10')) if base else '—'} "
            f"| {_fmt(o.get('p50_retrieval_ms'), 0)} "
            f"| {t['llm_calls'] if t else 0} |")
    return "\n".join(lines)


def per_class_table(row: dict) -> str:
    """Per-class breakdown - the table that carries the router result."""
    lines = ["| Query class | n | R@5 | R@20 | MRR | nDCG@10 |", "|---|---|---|---|---|---|"]
    for cls, m in row["by_class"].items():
        lines.append(
            f"| {cls} | {m['n_answerable']}/{m['n_queries']} "
            f"| {_fmt(m.get('recall@5'))} | {_fmt(m.get('recall@20'))} "
            f"| {_fmt(m.get('mrr'))} | {_fmt(m.get('ndcg@10'))} |"
        )
    return "\n".join(lines)
