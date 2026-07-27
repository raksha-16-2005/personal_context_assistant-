#!/usr/bin/env python
"""Turn built indices + the eval set into the README tables.

Every table in the README comes from here, and `make bench` reproduces all of
them from the published eval set. That reproducibility is the whole claim of
the project, so this script refuses to run against an eval set that does not
validate - a silently malformed label produces a number that looks fine.

    python scripts/run_ablation.py                       # every built config
    python scripts/run_ablation.py --dimension chunking  # one table
    python scripts/run_ablation.py --dimension rerank    # dimension 4
    python scripts/run_ablation.py --dimension transform # dimension 5

Indices are loaded once per (chunking, model) pair and reused across all four
retriever variants. Re-loading a model per row would dominate the runtime.

Dimensions 4 and 5 default to a *single* index config and a single retriever.
They are one-factor-at-a-time like everything else here, and crossing four
rerank arms with six indices and four retrievers would be 96 rows of which 90
answer no question the eval set is large enough to resolve.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation import harness as H  # noqa: E402
from emailrag.evaluation.evalset import load as load_eval, validate  # noqa: E402
from emailrag.index import chunktext as CT  # noqa: E402
from emailrag.index import embed as E  # noqa: E402
from emailrag.index import rerank as RR  # noqa: E402
from emailrag.index.dense import DenseIndex  # noqa: E402
from emailrag.index.sparse import SparseIndex  # noqa: E402
from emailrag.query.transform import QueryTransformer  # noqa: E402

# Dimensions 4 and 5 vary one thing against a fixed retriever. RRF is the
# baseline because it needs no tuned weight - see index/fusion.py.
PIVOT_RETRIEVER = "hybrid_rrf"


def discover(index_root: Path) -> list[Path]:
    """Built configs only - a directory without config.json is a partial run."""
    return sorted(d for d in index_root.iterdir()
                  if d.is_dir() and (d / "config.json").exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/queries.jsonl", type=Path)
    ap.add_argument("--sample", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--index-root", default="data/index", type=Path)
    ap.add_argument("--out", default="runs", type=Path)
    ap.add_argument("--retrievers", nargs="*", default=None)
    ap.add_argument("--dimension",
                    choices=["chunking", "model", "retriever", "rerank", "transform"],
                    default="retriever", help="which axis to group the printed table by")
    ap.add_argument("--sweep-weights", action="store_true",
                    help="also sweep the weighted-fusion weight (footnote only)")
    ap.add_argument("--rerank", nargs="*", default=None,
                    help=f"dimension-4 arms; default {list(RR.DEFAULT_ARMS)}")
    ap.add_argument("--transform", nargs="*", default=None,
                    help=f"dimension-5 arms; default {list(QueryTransformer.KINDS)}")
    ap.add_argument("--config", default=None,
                    help="restrict to one index config directory name (dimensions "
                         "4 and 5 default to whichever config is built)")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    # Dimensions 4 and 5 pivot on one retriever; the rest sweep all four.
    if args.retrievers is None:
        args.retrievers = ([PIVOT_RETRIEVER] if args.dimension in ("rerank", "transform")
                           else list(H.RETRIEVERS))
    rerank_arms = args.rerank if args.rerank is not None else (
        list(RR.DEFAULT_ARMS) if args.dimension == "rerank" else ["none"])
    transform_arms = args.transform if args.transform is not None else (
        list(QueryTransformer.KINDS) if args.dimension == "transform" else ["none"])

    for arm in rerank_arms:
        if arm not in RR.SPECS:
            print(f"error: unknown rerank arm {arm!r}; have {sorted(RR.SPECS)}",
                  file=sys.stderr)
            return 1
    for arm in transform_arms:
        if arm not in QueryTransformer.KINDS:
            print(f"error: unknown transform {arm!r}; have "
                  f"{list(QueryTransformer.KINDS)}", file=sys.stderr)
            return 1

    if not args.eval.exists():
        print(f"error: {args.eval} missing - label some queries first:\n"
              f"  .venv/bin/python scripts/verify_eval.py --index <config dir>",
              file=sys.stderr)
        return 1

    queries = [q for q in load_eval(args.eval) if q.verified]
    if not queries:
        print("error: no verified queries in the eval set", file=sys.stderr)
        return 1

    # Validate against the ids actually in the corpus. A label pointing at a
    # message outside the index caps recall below 1.0 and is invisible in the
    # aggregate numbers.
    if not args.skip_validation:
        corpus_ids = set(pq.read_table(args.sample, columns=["dedup_key"])
                         .column("dedup_key").to_pylist())
        report = validate(queries, corpus_ids)
        print(f"eval set: {len(queries)} verified queries")
        print(report.render())
        if not report.ok:
            print("\nrefusing to run against an invalid eval set "
                  "(--skip-validation to override)", file=sys.stderr)
            return 1

    configs = discover(args.index_root)
    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"error: no built index named {args.config!r}", file=sys.stderr)
            return 1
    elif args.dimension in ("rerank", "transform") and len(configs) > 1:
        # These dimensions vary one factor against a fixed index. Without a
        # stated config the choice would be alphabetical, which is not a
        # decision anyone made.
        print(f"error: --dimension {args.dimension} needs one index config; "
              f"{len(configs)} are built. Pass --config <name>, e.g.\n"
              f"  --config {configs[0].name}", file=sys.stderr)
        return 1
    if not configs:
        print(f"error: no built indices under {args.index_root}", file=sys.stderr)
        return 1
    print(f"\n{len(configs)} built config(s), {len(args.retrievers)} retriever(s), "
          f"rerank={rerank_arms}, transform={transform_arms}\n")

    E.configure_threads(6)
    rows: list[dict] = []
    sweeps: list[dict] = []

    for cfg_dir in configs:
        meta = json.loads((cfg_dir / "config.json").read_text())
        print(f"[{cfg_dir.name}] loading ...")
        bm25 = SparseIndex.load(cfg_dir / "bm25")
        dense = DenseIndex.load(cfg_dir / "dense")
        model = E.load_model(meta["model"])
        instruction = E.QUERY_INSTRUCTION.get(meta["model"], "")
        texts_cache: dict[str, object] = {}

        for retriever in args.retrievers:
            for transform_arm in transform_arms:
                for rerank_arm in rerank_arms:
                    run_cfg = H.RunConfig(meta["chunking"], meta["model"], retriever,
                                          rerank=rerank_arm, transform=transform_arm)
                    spec = RR.SPECS[rerank_arm]
                    transformer = (QueryTransformer(transform_arm)
                                   if transform_arm != "none" else None)

                    reranker = texts = None
                    if spec is not None:
                        # Whole-config texts, cached on disk and in memory. Which
                        # chunks land in a top-k depends on the retriever and the
                        # transform, so scoping the rebuild to "the ids this arm
                        # needs" would mean rebuilding per arm - or worse,
                        # silently dropping chunks a later arm retrieved but an
                        # earlier one did not.
                        if not texts_cache:
                            texts_cache["all"] = CT.load_or_build(
                                cfg_dir, args.sample, meta["chunking"])
                        texts = texts_cache["all"]
                        reranker = RR.CrossEncoderReranker.from_spec(spec)

                    label = arm_label(retriever, transform_arm, rerank_arm)
                    row = H.run(run_cfg, queries, bm25, dense, model, instruction,
                                transformer=transformer, reranker=reranker,
                                rerank_top_k=spec.top_k if spec else 0, texts=texts)
                    row["config"]["n_chunks"] = meta["n_chunks"]
                    row["config"]["chunks_per_message"] = meta["chunks_per_message"]
                    row["config"]["index"] = cfg_dir.name
                    if spec is not None:
                        row["config"]["rerank_spec"] = {
                            "model_id": spec.model_id, "top_k": spec.top_k,
                            "max_length": spec.max_length, "onnx_int8": spec.onnx_int8,
                            "backend": reranker.backend}
                    rows.append(row)
                    print(f"  {label:34s} " + summarize(row))
                    del reranker

        if args.sweep_weights:
            print(f"  sweeping fusion weight ...")
            sweeps.extend(H.sweep_weights(
                H.RunConfig(meta["chunking"], meta["model"], "hybrid_weighted"),
                queries, bm25, dense, model, instruction))

        del bm25, dense, model      # 16 GB budget - do not hold two models

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_utc": stamp,
        "n_queries": len(queries),
        "retrieve_depth": H.RETRIEVE_DEPTH,
        "dimension": args.dimension,
        "llm_cache": cache_line(),
        "rows": rows,
        "weight_sweep": sweeps,
    }
    json_path = args.out / f"ablation_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    table = (H.rerank_table(rows) if args.dimension == "rerank" else
             H.transform_table(rows) if args.dimension == "transform" else
             H.markdown_table(rows, group_by=args.dimension))

    print(f"\n{'='*70}")
    print(table)
    if len(rows) == 1 or args.dimension == "retriever":
        print(f"\nper-class breakdown ({rows[0]['config']['retriever']}):")
        print(per_class_or_note(rows))
    print(f"{'='*70}")
    if args.dimension in ("rerank", "transform"):
        print("latency is CPU-specific - see docs/HARDWARE.md for the machine")
    if args.transform or args.dimension == "transform":
        print(cache_line())

    md_path = args.out / f"ablation_{stamp}.md"
    md_path.write_text(table + "\n")
    print(f"\nwrote {json_path}\n      {md_path}")
    return 0


def arm_label(retriever: str, transform_arm: str, rerank_arm: str) -> str:
    parts = [retriever]
    if transform_arm != "none":
        parts.append(transform_arm)
    if rerank_arm != "none":
        parts.append(rerank_arm)
    return " + ".join(parts)


def summarize(row: dict) -> str:
    o = row["overall"]
    out = (f"R@5={o.get('recall@5', 0):.3f} R@20={o.get('recall@20', 0):.3f} "
           f"nDCG@10={o.get('ndcg@10', 0):.3f} p95={o.get('p95_ms', 0):.0f}ms")
    if o.get("p50_rerank_ms"):
        out += f" (rerank {o['p50_rerank_ms']:.0f}ms)"
    t = row.get("transform")
    if t:
        out += f" [{t['fired']}/{t['queries']} fired"
        if t["degraded"]:
            out += f", {t['degraded']} degraded"
        out += "]"
    return out


def cache_line() -> str:
    from emailrag.llm.client import default_cache
    return default_cache().stats.render()


def per_class_or_note(rows: list[dict]) -> str:
    best = max(rows, key=lambda r: r["overall"].get("ndcg@10", 0))
    return H.per_class_table(best)


if __name__ == "__main__":
    raise SystemExit(main())
