#!/usr/bin/env python
"""Dimension 6: classify every recall@20 miss and print the taxonomy.

    make failures                      # newest run, best row
    python scripts/analyze_failures.py --run runs/ablation_....json --list

Reads a results file `run_ablation.py` already wrote - no retrieval is re-run, so
re-classifying costs nothing and the taxonomy is reproducible from the same
artifact as the tables.

The output has two jobs. The counts become the README's "Where this fails"
section. The `--list` output is a worklist: every miss with the evidence for its
category, so the ones that are actually bad labels can be found and fixed. That
second job is why `bad_label` is never assigned automatically - see
evaluation/failures.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation import failures as F  # noqa: E402
from emailrag.evaluation.evalset import load as load_eval  # noqa: E402
from emailrag.index import chunktext as CT  # noqa: E402


def newest_run(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.glob("ablation_*.json"))
    return candidates[-1] if candidates else None


def pick_row(payload: dict, index: str | None, retriever: str | None) -> dict:
    """Default to the best-nDCG row: the taxonomy of the *winning* config is
    what the README reports, not the taxonomy of an arm nobody would ship."""
    rows = payload["rows"]
    if index:
        rows = [r for r in rows if r["config"].get("index") == index]
    if retriever:
        rows = [r for r in rows if r["config"]["retriever"] == retriever]
    if not rows:
        raise SystemExit("no rows match those filters")
    return max(rows, key=lambda r: r["overall"].get("ndcg@10", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None,
                    help="results file; defaults to the newest in runs/")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--index-root", type=Path, default=Path("data/index"))
    ap.add_argument("--index", default=None, help="restrict to one index config")
    ap.add_argument("--retriever", default=None)
    ap.add_argument("-k", type=int, default=20, help="the recall cutoff being explained")
    ap.add_argument("--list", action="store_true",
                    help="print every miss with its evidence, as a worklist")
    ap.add_argument("--no-text", action="store_true",
                    help="skip loading chunk texts (term overlap will read 0)")
    args = ap.parse_args()

    run_path = args.run or newest_run(args.runs_dir)
    if run_path is None or not run_path.exists():
        print(f"error: no results file (looked in {args.runs_dir}). "
              f"Run `make bench` first.", file=sys.stderr)
        return 1
    if not args.eval.exists():
        print(f"error: {args.eval} missing - label some queries first", file=sys.stderr)
        return 1

    payload = json.loads(run_path.read_text())
    row = pick_row(payload, args.index, args.retriever)
    queries = {q.query_id: q for q in load_eval(args.eval)}

    print(f"{run_path.name}")
    print(f"  row: {row['config'].get('index', '?')} | "
          f"{row['config']['retriever']} | rerank={row['config'].get('rerank')} | "
          f"transform={row['config'].get('transform')}")
    print(f"  nDCG@10={row['overall'].get('ndcg@10', 0):.3f} "
          f"recall@{args.k}={row['overall'].get(f'recall@{args.k}', 0):.3f}\n")

    # Message text, for the term-overlap measurement that keeps "vocabulary
    # mismatch" a number rather than a shrug. Optional: with --no-text the
    # category is still assigned, the evidence is just absent.
    texts_by_message: dict[str, str] = {}
    if not args.no_text:
        index_name = row["config"].get("index")
        index_dir = args.index_root / index_name if index_name else None
        if index_dir and (index_dir / "chunks.jsonl").exists():
            chunk_texts = CT.load_or_build(index_dir, args.sample,
                                           row["config"]["chunking"])
            meta = CT.load_chunk_meta(index_dir / "chunks.jsonl")
            for chunk_id, text in chunk_texts.texts.items():
                key = meta.get(chunk_id, {}).get("dedup_key")
                if key:
                    texts_by_message[key] = texts_by_message.get(key, "") + " " + text
        else:
            print("  (no chunk texts available - term overlap will read 0)\n")

    chunk_ranks = {r["query_id"]: [(cid, rank) for cid, rank in
                                   r.get("relevant_chunk_ranks", [])]
                   for r in row["per_query"]}

    report = F.classify(row["per_query"], queries, chunk_ranks, texts_by_message,
                        k=args.k)

    print(report.render())

    if args.list:
        print(f"\n{'-'*70}\nworklist - every miss, with the evidence for its category")
        for miss in sorted(report.misses, key=lambda m: m.category):
            print(f"\n[{miss.category}] {miss.query_id}  {miss.query}")
            print(f"  class={miss.query_class} labelled={miss.n_relevant} "
                  f"best_message_rank={miss.best_rank} "
                  f"best_chunk_rank={miss.best_chunk_rank} "
                  f"term_overlap={miss.term_overlap:.2f}")
            print(f"  {miss.detail}")
        print(f"\nTo disown a label, put 'bad_label: <why>' in that query's "
              f"`notes` in {args.eval} and re-run. Nothing here marks a label bad "
              f"on its own - see evaluation/failures.py.")

    out = args.runs_dir / f"failures_{run_path.stem.replace('ablation_', '')}.json"
    out.write_text(json.dumps({
        "run": run_path.name,
        "row": row["config"],
        "k": args.k,
        "n_answerable": report.n_answerable,
        "counts": dict(report.counts),
        # asdict, not vars: Miss is a slots dataclass and has no __dict__.
        "misses": [dataclasses.asdict(m) for m in report.misses],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
