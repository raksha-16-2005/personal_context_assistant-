#!/usr/bin/env python
"""Load a built index into pgvector, then measure what the approximation costs.

    python scripts/load_pgvector.py --config thread_aware__all-MiniLM-L6-v2
    python scripts/load_pgvector.py --config ... --measure-recall

`index/store.py` has existed since week 0 and nothing called it. This is the
caller.

**The recall measurement is the point, not the load.** Every ablation number in
this project comes from exact search, on purpose: HNSW recall loss varies with
dimensionality and with how clustered a model's embedding space is, so an
HNSW-backed comparison of bge-base against MiniLM would measure the model *and*
the index while reporting it as a property of the model. The served system needs
the ANN index for latency, which means the loss is real and has to be quantified
rather than absorbed. `--measure-recall` computes recall@k of the HNSW ranking
against the exact ranking over the same queries, which turns a hidden confound
into a reported number.

Recall here means *agreement with exact search*, not relevance. A chunk the exact
search missed cannot be recovered by an approximate one, so this measures only
what the index gives up - and that is the only thing it can honestly measure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index import chunktext as CT  # noqa: E402
from emailrag.index import embed as E  # noqa: E402
from emailrag.index import store as S  # noqa: E402
from emailrag.index.dense import DenseIndex  # noqa: E402

# ef_search values to sweep. This is the knob a deployment actually turns: higher
# is more accurate and slower, and where to sit on that curve is a decision the
# numbers should inform rather than a default nobody chose.
EF_SEARCH_SWEEP = (40, 100, 200, 400)

DEFAULT_DSN = "postgresql:///emailrag"


def load_config(index_dir: Path) -> dict:
    path = index_dir / "config.json"
    if not path.exists():
        raise SystemExit(f"error: {path} missing - is {index_dir} a built index?")
    return json.loads(path.read_text())


def do_load(conn, table: str, index_dir: Path, sample: Path, meta: dict,
            drop: bool) -> int:
    """Chunk texts plus vectors into one table.

    The chunks are rebuilt from the corpus rather than read from the index, because
    the index stores no text (see index/chunktext.py) and the table needs it: the
    served system shows excerpts, and the reranker scores them. The rebuild is
    verified against the index's own chunk ids, so a mismatch fails here instead of
    loading vectors paired with the wrong passages.
    """
    print(f"rebuilding chunks for parity ({meta['chunking']}) ...")
    chunks = CT.rebuild_chunks(sample, meta["chunking"])

    dense = DenseIndex.load(index_dir / "dense")
    CT.verify_parity([c.chunk_id for c in chunks], list(dense.chunk_ids),
                     where=f"{index_dir.name} (rebuild vs dense index)", ordered=True)
    CT.verify_normalized(dense.matrix)
    print(f"  parity: {len(chunks):,} chunk ids match, in order  OK")

    S.create_table(conn, table, dim=dense.dim, drop=drop)
    print(f"COPY into {table} ({dense.dim}-dim) ...")
    t0 = time.time()
    n = S.copy_chunks(conn, table, chunks, dense.matrix)
    print(f"  {n:,} rows in {time.time()-t0:.0f}s")

    print("building HNSW (vector_ip_ops - vectors are normalized, so inner "
          "product ranks identically to cosine and skips the norm) ...")
    t0 = time.time()
    S.build_hnsw(conn, table)
    print(f"  built in {time.time()-t0:.0f}s")
    return n


def measure_recall(conn, table: str, index_dir: Path, meta: dict,
                   queries: list[str], top_k: int, ef_values: tuple[int, ...]) -> dict:
    """HNSW ranking vs the exact ranking, over the same queries.

    Uses the eval set's queries when available rather than random vectors: recall
    loss is not uniform over query space, and a synthetic query is not the
    distribution the system serves.
    """
    print(f"\nloading the exact index for the baseline ...")
    dense = DenseIndex.load(index_dir / "dense")
    E.configure_threads(6)
    model = E.load_model(meta["model"])
    instruction = E.QUERY_INSTRUCTION.get(meta["model"], "")
    qvecs = E.encode_queries(model, queries, instruction or None)

    print(f"exact search over {len(queries)} queries at top-{top_k} ...")
    exact: list[list[str]] = []
    exact_ms: list[float] = []
    for vec in qvecs:
        t0 = time.perf_counter()
        exact.append([cid for cid, _ in dense.search(vec, top_k=top_k)])
        exact_ms.append((time.perf_counter() - t0) * 1000)

    rows = []
    for ef in ef_values:
        overlaps, latencies, rank1 = [], [], 0
        for vec, gold in zip(qvecs, exact):
            t0 = time.perf_counter()
            got = [cid for cid, _ in S.search(conn, table, vec, top_k=top_k,
                                              ef_search=ef)]
            latencies.append((time.perf_counter() - t0) * 1000)
            overlaps.append(len(set(got) & set(gold)) / len(gold) if gold else 1.0)
            if got and gold and got[0] == gold[0]:
                rank1 += 1

        row = {
            "ef_search": ef,
            "recall_at_k": round(statistics.mean(overlaps), 4),
            "worst_query_recall": round(min(overlaps), 4),
            "top1_agreement": round(rank1 / len(queries), 4),
            "p50_ms": round(statistics.median(latencies), 2),
            "p95_ms": round(sorted(latencies)[min(int(len(latencies) * 0.95),
                                                  len(latencies) - 1)], 2),
        }
        rows.append(row)
        print(f"  ef_search={ef:4d}  recall@{top_k}={row['recall_at_k']:.4f}  "
              f"worst={row['worst_query_recall']:.4f}  "
              f"top1={row['top1_agreement']:.4f}  p50={row['p50_ms']:.1f}ms")

    return {
        "table": table,
        "index": index_dir.name,
        "model": meta["model"],
        "top_k": top_k,
        "n_queries": len(queries),
        "exact_p50_ms": round(statistics.median(exact_ms), 2),
        "note": ("recall here means agreement with exact search, not relevance - "
                 "a chunk exact search missed cannot be recovered by an "
                 "approximate one. Latency is CPU-specific; see docs/HARDWARE.md."),
        "rows": rows,
    }


def recall_table(report: dict) -> str:
    lines = [f"| ef_search | recall@{report['top_k']} vs exact | worst query "
             f"| top-1 agreement | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|"]
    for r in report["rows"]:
        lines.append(f"| {r['ef_search']} | {r['recall_at_k']:.4f} "
                     f"| {r['worst_query_recall']:.4f} | {r['top1_agreement']:.4f} "
                     f"| {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")
    lines.append(f"| *exact (brute force)* | 1.0000 | 1.0000 | 1.0000 "
                 f"| {report['exact_p50_ms']:.1f} | — |")
    return "\n".join(lines)


def eval_queries(path: Path, limit: int) -> list[str]:
    """Real queries beat synthetic ones - recall loss is not uniform over query
    space, and a random vector is not the distribution the system serves."""
    if path.exists():
        from emailrag.evaluation.evalset import load as load_eval
        queries = [q.query for q in load_eval(path) if q.verified][:limit]
        if queries:
            print(f"  using {len(queries)} verified eval queries")
            return queries
    print(f"  warning: no verified queries in {path}; falling back to generic "
          f"probes, which are not the served distribution", file=sys.stderr)
    return [
        "what was decided about the pricing model",
        "confidentiality agreement revisions",
        "gas nominations for next month",
        "who is handling the california filings",
        "counterparty credit review",
        "merger terms and exchange ratio",
        "capacity release posting",
        "legal review of the master agreement",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="index directory name under --index-root")
    ap.add_argument("--index-root", type=Path, default=Path("data/index"))
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--maintenance-work-mem", default="2GB",
                    help="raised per session only - this server hosts other projects")
    ap.add_argument("--drop", action="store_true", help="recreate the table")
    ap.add_argument("--skip-load", action="store_true",
                    help="measure against an already-loaded table")
    ap.add_argument("--measure-recall", action="store_true")
    ap.add_argument("--top-k", type=int, default=20,
                    help="the k recall is measured at; 20 matches recall@20")
    ap.add_argument("--max-queries", type=int, default=80)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    index_dir = args.index_root / args.config
    meta = load_config(index_dir)
    table = S.table_name(meta["chunking"], meta["model"])
    print(f"config {args.config}\n  table {table}\n  dsn   {args.dsn}")

    try:
        with S.connect(args.dsn, args.maintenance_work_mem) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            if not args.skip_load:
                do_load(conn, table, index_dir, args.sample, meta, args.drop)

            if not args.measure_recall:
                print("\nloaded. Add --measure-recall to quantify what HNSW gives "
                      "up against exact search.")
                return 0

            queries = eval_queries(args.eval, args.max_queries)
            report = measure_recall(conn, table, index_dir, meta, queries,
                                    args.top_k, EF_SEARCH_SWEEP)
    except Exception as exc:                            # noqa: BLE001
        import psycopg
        if isinstance(exc, psycopg.OperationalError):
            print(f"\nerror: cannot reach Postgres at {args.dsn}\n"
                  f"  {exc}\n"
                  f"  start it:  brew services start postgresql@15\n"
                  f"  create it: createdb emailrag\n"
                  f"  pgvector:  CREATE EXTENSION vector;", file=sys.stderr)
            return 1
        raise

    print(f"\n{'='*70}")
    print(recall_table(report))
    print(f"{'='*70}")
    print(report["note"])

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"hnsw_recall_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
