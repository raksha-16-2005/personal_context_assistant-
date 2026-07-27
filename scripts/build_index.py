#!/usr/bin/env python
"""Build one ablation config: chunk -> embed -> BM25, persisted to data/index/.

A "config" is a (chunking strategy, embedding model) pair. Each is built and
cached independently so the 4x4 matrix can be filled in over several nights
without redoing finished work, and so a crash in hour nine of a bge-base pass
costs one shard rather than the whole run.

    python scripts/build_index.py --chunking thread_aware --model BAAI/bge-small-en-v1.5

Cost on this machine, at the 50k-message sample (measured, docs/s):
    all-MiniLM-L6-v2   67.2  ->  ~15 min
    bge-small-en-v1.5  34.5  ->  ~30 min
    bge-base-en-v1.5    9.8  ->  ~1.7 h
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.chunking import strategies as S  # noqa: E402
from emailrag.index import embed as E  # noqa: E402
from emailrag.index.dense import DenseIndex  # noqa: E402
from emailrag.index.sparse import SparseIndex  # noqa: E402


def config_dir(root: Path, chunking: str, model_id: str) -> Path:
    return root / f"{chunking}__{model_id.split('/')[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--index-root", default="data/index", type=Path)
    ap.add_argument("--chunking", required=True, choices=sorted(S.STRATEGIES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    if not args.sample.exists():
        print(f"error: {args.sample} missing - run `make sample` first", file=sys.stderr)
        return 1

    out = config_dir(args.index_root, args.chunking, args.model)
    done_marker = out / "config.json"
    if done_marker.exists() and not args.force:
        print(f"cached: {out} (use --force to rebuild)")
        return 0

    E.configure_threads(args.threads)
    out.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.sample} ...")
    messages = pq.read_table(args.sample).to_pylist()
    print(f"  {len(messages):,} messages")

    # One reference tokenizer for all strategies and models - chunk boundaries
    # must not shift when the embedding model changes, or dimensions 1 and 2
    # stop being independent.
    from transformers import AutoTokenizer
    print(f"chunking: {args.chunking} (reference tokenizer {S.REFERENCE_TOKENIZER})")
    tokenizer = AutoTokenizer.from_pretrained(S.REFERENCE_TOKENIZER)

    t0 = time.time()
    chunks = S.chunk_corpus(args.chunking, tokenizer, messages)
    chunk_secs = time.time() - t0
    if not chunks:
        print("error: chunking produced nothing", file=sys.stderr)
        return 1

    n_tok = [c.n_tokens for c in chunks]
    print(f"  {len(chunks):,} chunks in {chunk_secs:.0f}s "
          f"({len(chunks)/len(messages):.2f} per message, "
          f"mean {sum(n_tok)/len(n_tok):.0f} tokens, max {max(n_tok)})")

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]

    print(f"BM25 ...")
    t0 = time.time()
    SparseIndex.build(ids, texts).save(out / "bm25")
    print(f"  built in {time.time()-t0:.0f}s")

    print(f"embedding with {args.model} ...")
    model = E.load_model(args.model)
    t0 = time.time()
    vectors = E.encode_corpus(model, texts, out / "shards", batch_size=args.batch_size)
    embed_secs = time.time() - t0
    rate = len(texts) / embed_secs if embed_secs else 0
    print(f"  {len(texts):,} chunks in {embed_secs/60:.1f} min ({rate:.1f} docs/s)")

    DenseIndex(ids, vectors).save(out / "dense")

    # Chunk metadata lives beside the vectors so the harness can map a hit back
    # to its message without re-reading the corpus.
    with open(out / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "dedup_key": c.dedup_key,
                                 "thread_id": c.thread_id, "n_tokens": c.n_tokens}) + "\n")

    done_marker.write_text(json.dumps({
        "chunking": args.chunking,
        "model": args.model,
        "dim": int(vectors.shape[1]),
        "n_messages": len(messages),
        "n_chunks": len(chunks),
        "chunks_per_message": round(len(chunks) / len(messages), 3),
        "mean_tokens": round(sum(n_tok) / len(n_tok), 1),
        "embed_seconds": round(embed_secs, 1),
        "embed_docs_per_sec": round(rate, 2),
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
