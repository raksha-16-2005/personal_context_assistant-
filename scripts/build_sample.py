#!/usr/bin/env python
"""Thread reconstruction over the full corpus, then the ablation subsample.

Threading reads only the nine columns it needs rather than the whole table -
bodies dominate the Parquet footprint and are irrelevant to who-replied-to-whom.
Sampling then works on thread ids alone, and only the final selected rows are
materialised with their bodies. Peak RSS stays well inside the 16 GB budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.corpus.sample import stratified_thread_sample  # noqa: E402
from emailrag.corpus.threads import assign_threads  # noqa: E402

THREAD_COLS = ["dedup_key", "message_id", "in_reply_to", "references",
               "subject_norm", "sender", "recipients", "cc", "date_utc"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", default="data/interim/messages.parquet", type=Path)
    ap.add_argument("--out", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    if not args.messages.exists():
        print(f"error: {args.messages} missing - run `make corpus` first", file=sys.stderr)
        return 1

    print(f"reading threading columns from {args.messages} ...")
    light = pq.read_table(args.messages, columns=THREAD_COLS).to_pylist()
    print(f"  {len(light):,} messages")

    print("reconstructing threads ...")
    thread_ids, tstats = assign_threads(light)
    for k, v in tstats.items():
        print(f"  {k:24s} {v:,}" if isinstance(v, int) else f"  {k:24s} {v}")

    for row, tid in zip(light, thread_ids):
        row["thread_id"] = tid

    print(f"\nsampling ~{args.target:,} messages (whole threads, stratified) ...")
    sampled, sstats = stratified_thread_sample(light, args.target, args.seed)
    for k, v in sstats.items():
        print(f"  {k:24s} {v:,}" if isinstance(v, int) else f"  {k:24s} {v}")

    # Materialise the selected rows with bodies, and carry thread_id across.
    keep = set(m["dedup_key"] for m in sampled)
    tid_by_key = {m["dedup_key"]: m["thread_id"] for m in sampled}
    if not keep:
        print("error: sample is empty - is the corpus empty?", file=sys.stderr)
        return 1

    full = pq.read_table(args.messages)
    # The type is explicit: an untyped empty pa.array() infers null and
    # is_in then fails with "string vs null" instead of matching nothing.
    mask = pc.is_in(full.column("dedup_key"),
                    value_set=pa.array(sorted(keep), type=pa.string()))
    subset = full.filter(mask)
    subset = subset.append_column(
        "thread_id",
        pa.array([tid_by_key[k] for k in subset.column("dedup_key").to_pylist()], pa.string()),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, args.out, compression="zstd")

    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps({"threading": tstats, "sampling": sstats,
                                      "seed": args.seed}, indent=2))
    print(f"\nwrote {subset.num_rows:,} rows -> {args.out} "
          f"({args.out.stat().st_size/1e6:.0f} MB)")
    print(f"stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
