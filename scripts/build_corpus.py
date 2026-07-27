#!/usr/bin/env python
"""maildir -> deduplicated, bulk-filtered Parquet corpus.

Streams: workers parse one maildir user each, the parent deduplicates and
flushes batches straight to disk. Only the set of seen dedup keys is held in
memory (~100 MB at 500k messages), so peak RSS stays flat and this runs
comfortably alongside everything else on a 16 GB machine.

Uses ordered imap (not imap_unordered) so the surviving copy of a duplicated
message is always the same one - the run must be reproducible.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.corpus import filters  # noqa: E402
from emailrag.corpus.enron import iter_user_dirs, parse_user_dir  # noqa: E402

SCHEMA = pa.schema([
    ("message_id", pa.string()),
    ("dedup_key", pa.string()),
    ("date_utc", pa.timestamp("us", tz="UTC")),
    ("sender", pa.string()),
    ("recipients", pa.string()),
    ("cc", pa.string()),
    ("subject", pa.string()),
    ("subject_norm", pa.string()),
    ("body", pa.string()),
    ("body_new", pa.string()),
    ("in_reply_to", pa.string()),
    ("references", pa.string()),
    ("has_list_unsubscribe", pa.bool_()),
    ("source_path", pa.string()),
    ("owner", pa.string()),
    ("folder", pa.string()),
])

FLUSH_EVERY = 25_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maildir", default="data/raw/maildir", type=Path)
    ap.add_argument("--out", default="data/interim/messages.parquet", type=Path)
    ap.add_argument("--workers", type=int, default=5,
                    help="5 of 6 physical cores; leaves one for the writer")
    ap.add_argument("--keep-bulk", action="store_true",
                    help="skip the rules filter (for measuring what it removes)")
    args = ap.parse_args()

    if not args.maildir.is_dir():
        print(f"error: {args.maildir} not found - run `make download` first", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    user_dirs = iter_user_dirs(args.maildir)
    print(f"{len(user_dirs)} maildir users -> {args.out}")

    seen: set[str] = set()
    n_parsed = n_dupe = n_bulk = n_kept = 0
    buffer: list[dict] = []
    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")
    t0 = time.time()

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        cols = {name: [row[name] for row in buffer] for name in SCHEMA.names}
        writer.write_table(pa.table(cols, schema=SCHEMA))
        buffer = []

    try:
        with mp.Pool(args.workers) as pool:
            tasks = [(d, args.maildir) for d in user_dirs]
            for messages in tqdm(pool.imap(parse_user_dir, tasks), total=len(tasks), unit="user"):
                for msg in messages:
                    n_parsed += 1
                    if msg["dedup_key"] in seen:
                        n_dupe += 1
                        continue
                    seen.add(msg["dedup_key"])
                    if not args.keep_bulk and filters.is_bulk(msg):
                        n_bulk += 1
                        continue
                    buffer.append(msg)
                    n_kept += 1
                if len(buffer) >= FLUSH_EVERY:
                    flush()
        flush()
    finally:
        writer.close()

    unique = n_parsed - n_dupe
    elapsed = time.time() - t0
    print(f"\nparsed        {n_parsed:>9,} files in {elapsed/60:.1f} min")
    print(f"duplicates    {n_dupe:>9,}  ({n_dupe/max(n_parsed,1)*100:.1f}% of files)")
    print(f"unique msgs   {unique:>9,}")
    print(filters.report(unique, n_bulk))

    # Second pass, corpus-level: drop recurring (sender, subject) blasts. This
    # cannot be done in the streaming loop above - it needs counts over the
    # whole corpus - and it matters because these blasts wreck thread
    # reconstruction, not just index size.
    if not args.keep_bulk:
        n_kept = _drop_recurring_blasts(args.out)

    print(f"kept          {n_kept:>9,}  -> {args.out} "
          f"({args.out.stat().st_size/1e6:.0f} MB)")
    return 0


def _drop_recurring_blasts(path: Path) -> int:
    table = pq.read_table(path)
    pairs = list(zip(table.column("sender").to_pylist(),
                     table.column("subject_norm").to_pylist()))
    blasts = filters.find_recurring_blasts(pairs)
    if not blasts:
        print("recurring blasts: none found")
        return table.num_rows

    mask = pa.array([pair not in blasts for pair in pairs], pa.bool_())
    filtered = table.filter(mask)
    dropped = table.num_rows - filtered.num_rows
    print(f"recurring blasts: dropped {dropped:,} messages across "
          f"{len(blasts):,} (sender, subject) pairs "
          f"(>= {filters.RECURRING_THRESHOLD} repeats)")
    pq.write_table(filtered, path, compression="zstd")
    return filtered.num_rows


if __name__ == "__main__":
    raise SystemExit(main())
