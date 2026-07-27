#!/usr/bin/env python
"""Validate the published eval set. Runs in CI and inside `make bench`.

Exits non-zero on any error so a malformed label cannot reach a results table.
Warnings (stratification shortfalls, unverified queries, near-duplicates) are
reported but do not fail - they are expected mid-labelling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation.evalset import (  # noqa: E402
    TARGET_COUNTS, ValidationError, load, stratification, validate,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval/queries.jsonl", type=Path)
    ap.add_argument("--sample", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()

    if not args.eval.exists():
        print(f"error: {args.eval} does not exist", file=sys.stderr)
        return 1

    try:
        queries = load(args.eval)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    corpus_ids = None
    if args.sample.exists():
        corpus_ids = set(pq.read_table(args.sample, columns=["dedup_key"])
                         .column("dedup_key").to_pylist())
        print(f"checking labels against {len(corpus_ids):,} corpus messages")
    else:
        print(f"warning: {args.sample} missing - cannot check that labels "
              f"point at messages that exist")

    report = validate(queries, corpus_ids)
    verified = sum(1 for q in queries if q.verified)

    print(f"\n{len(queries)} queries ({verified} verified)")
    counts = stratification(queries)
    for cls, target in TARGET_COUNTS.items():
        mark = "ok" if counts.get(cls, 0) == target else ".."
        print(f"  {mark} {cls:14s} {counts.get(cls, 0):3d} / {target}")

    labels = sum(len(q.relevant_message_ids) for q in queries)
    answerable = [q for q in queries if q.answerable and q.relevant_message_ids]
    if answerable:
        print(f"\n{labels} relevance labels, "
              f"{labels/len(answerable):.1f} per answerable query")

    print(f"\nvalidation:")
    print(report.render())

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        print("\nstrict mode: warnings are errors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
