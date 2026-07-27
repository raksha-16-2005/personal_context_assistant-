#!/usr/bin/env python
"""Import GPU-built embeddings into data/index/ so `make bench` runs unchanged.

The companion to `notebooks/kaggle_build_indices.ipynb`. The notebook chunks and
embeds on a free T4; this script drops the result into the local index layout,
builds the BM25 side (CPU-cheap, and not worth a GPU session), and refuses to
accept anything it cannot prove came from the same corpus.

    python scripts/import_embeddings.py --from ~/Downloads/kaggle-emailrag-indices

**The parity check is the point of this script.** A `.npy` matrix carries no
evidence about which chunks it describes. If the notebook chunked a different
sample, a different strategy, or the same strategy with a drifted tokenizer, the
vectors still load, still search, and still produce plausible numbers - for the
wrong passages. So every import re-chunks the local corpus and requires the chunk
ids to match the remote ones exactly, in order, because row i of the matrix is
chunk i and nothing else records that pairing.

**Embed throughput is not imported as a local measurement.** Quality metrics
(recall, MRR, nDCG) are hardware-independent: same weights, same math, same
answer on a T4 or a laptop. Embedding *speed* is not, so the platform it was
measured on is recorded in config.json and `embed_docs_per_sec` from a GPU run
never appears in a table of CPU numbers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.chunking import strategies as S  # noqa: E402
from emailrag.index import chunktext as CT  # noqa: E402
from emailrag.index.dense import DenseIndex  # noqa: E402
from emailrag.index.sparse import SparseIndex  # noqa: E402

REQUIRED = ("config.json", "dense/embeddings.npy", "dense/chunk_ids.txt")


def discover(root: Path) -> list[Path]:
    """Config directories in a downloaded Kaggle output."""
    out = []
    for path in sorted(root.rglob("config.json")):
        d = path.parent
        if all((d / r).exists() for r in REQUIRED):
            out.append(d)
    return out


def local_chunks(sample: Path, chunking: str, tokenizer):
    t0 = time.time()
    chunks = CT.rebuild_chunks(sample, chunking, tokenizer)
    print(f"  local rechunk: {len(chunks):,} chunks in {time.time()-t0:.0f}s")
    return chunks


def import_one(src: Path, index_root: Path, sample: Path, tokenizer,
               force: bool = False) -> bool:
    meta = json.loads((src / "config.json").read_text())
    chunking, model_id = meta["chunking"], meta["model"]
    name = f"{chunking}__{model_id.split('/')[-1]}"
    dest = index_root / name

    print(f"\n[{name}]  from {src}")
    if (dest / "config.json").exists() and not force:
        print(f"  already built locally at {dest} (use --force to overwrite)")
        return False

    remote_ids = (src / "dense" / "chunk_ids.txt").read_text().splitlines()
    vectors = np.load(src / "dense" / "embeddings.npy")
    print(f"  remote: {len(remote_ids):,} ids, matrix {vectors.shape}, "
          f"{vectors.dtype}")

    if vectors.shape[0] != len(remote_ids):
        print(f"  ERROR: {vectors.shape[0]:,} vectors vs {len(remote_ids):,} ids",
              file=sys.stderr)
        return False

    chunks = local_chunks(sample, chunking, tokenizer)
    CT.verify_parity([c.chunk_id for c in chunks], remote_ids,
                     where=f"{name} (local rechunk vs notebook)", ordered=True)
    print(f"  parity: {len(chunks):,} chunk ids match, in order  OK")

    # Normalisation is assumed by DenseIndex.search - the dot product *is* the
    # cosine only if the rows are unit length. A notebook that forgot
    # normalize_embeddings would otherwise silently change what "similarity"
    # means, so it is checked rather than trusted.
    norms = np.linalg.norm(vectors[:min(1000, len(vectors))], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        print(f"  ERROR: vectors are not L2-normalized (norms "
              f"{norms.min():.4f}-{norms.max():.4f}). DenseIndex treats the dot "
              f"product as the cosine; re-embed with normalize_embeddings=True.",
              file=sys.stderr)
        return False
    print(f"  norms: unit length  OK")

    dest.mkdir(parents=True, exist_ok=True)
    DenseIndex([c.chunk_id for c in chunks], vectors).save(dest / "dense")

    print(f"  BM25 (built locally - it is CPU-cheap and needs no GPU session) ...")
    t0 = time.time()
    SparseIndex.build([c.chunk_id for c in chunks],
                      [c.text for c in chunks]).save(dest / "bm25")
    print(f"    built in {time.time()-t0:.0f}s")

    with open(dest / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "dedup_key": c.dedup_key,
                                 "thread_id": c.thread_id, "n_tokens": c.n_tokens}) + "\n")

    n_tok = [c.n_tokens for c in chunks]
    (dest / "config.json").write_text(json.dumps({
        "chunking": chunking,
        "model": model_id,
        "dim": int(vectors.shape[1]),
        "n_messages": meta.get("n_messages"),
        "n_chunks": len(chunks),
        "chunks_per_message": round(len(chunks) / meta["n_messages"], 3)
        if meta.get("n_messages") else None,
        "mean_tokens": round(sum(n_tok) / len(n_tok), 1),
        # Provenance. `embed_platform` exists so a GPU throughput figure can
        # never be pasted into a table of CPU numbers: quality is portable,
        # speed is not.
        "embed_platform": meta.get("embed_platform", "unknown-gpu"),
        "embed_seconds": meta.get("embed_seconds"),
        "embed_docs_per_sec": meta.get("embed_docs_per_sec"),
        "embed_torch": meta.get("torch"),
        "imported_from": str(src),
        "parity_verified": True,
    }, indent=2))
    print(f"  wrote {dest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="downloaded Kaggle output directory")
    ap.add_argument("--sample", default=Path("data/interim/sample.parquet"), type=Path)
    ap.add_argument("--index-root", default=Path("data/index"), type=Path)
    ap.add_argument("--config", default=None, help="import only this chunking__model")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.sample.exists():
        print(f"error: {args.sample} missing - the parity check re-chunks the "
              f"local corpus, so `make sample` has to have run", file=sys.stderr)
        return 1

    configs = discover(args.src)
    if args.config:
        configs = [c for c in configs
                   if json.loads((c / "config.json").read_text())["chunking"]
                   in args.config or args.config in c.name]
    if not configs:
        print(f"error: nothing importable under {args.src}. Expected directories "
              f"containing {', '.join(REQUIRED)}", file=sys.stderr)
        return 1

    print(f"{len(configs)} config(s) to import")
    # One tokenizer for every config - the same reference tokenizer the local
    # build uses, which is what makes the parity check meaningful.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(S.REFERENCE_TOKENIZER)

    imported = 0
    for src in configs:
        try:
            imported += bool(import_one(src, args.index_root, args.sample,
                                        tokenizer, args.force))
        except CT.ParityError as exc:
            print(f"  PARITY FAILURE - refusing to import:\n    {exc}",
                  file=sys.stderr)

    print(f"\n{imported} config(s) imported into {args.index_root}")
    if imported:
        print("Quality metrics from these are directly comparable to locally-built "
              "ones.\nEmbed throughput is not - see `embed_platform` in each "
              "config.json.")
    return 0 if imported else 1


if __name__ == "__main__":
    raise SystemExit(main())
