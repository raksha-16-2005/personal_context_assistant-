"""Batched CPU embedding.

Thread count is pinned to 6 (physical cores). Letting torch use all 12 logical
threads measurably *reduces* throughput here - the hyperthread pairs contend
for the same AVX2 units, and these are pure GEMM workloads.

Encoding is checkpointed to disk in shards. A full-corpus bge-base pass is a
~10 hour job on this machine; losing it to an OOM or an accidental Ctrl-C in
hour nine is not acceptable, and resuming from the last shard costs nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

SHARD_SIZE = 20_000


def configure_threads(n: int = 6) -> None:
    torch.set_num_threads(n)
    # Intra-op parallelism is where the win is; inter-op adds contention.
    torch.set_num_interop_threads(1)


def load_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device="cpu")


def encode_corpus(
    model,
    texts: list[str],
    out_dir: Path,
    batch_size: int = 32,
    normalize: bool = True,
    resume: bool = True,
) -> np.ndarray:
    """Encode `texts`, checkpointing every SHARD_SIZE rows into `out_dir`.

    Embeddings are L2-normalized so cosine similarity is a plain dot product,
    which lets pgvector use the cheaper inner-product operator.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    n = len(texts)
    dim = model.get_sentence_embedding_dimension()

    shards: list[np.ndarray] = []
    start = 0
    if resume:
        while (out_dir / f"shard_{start:07d}.npy").exists():
            shards.append(np.load(out_dir / f"shard_{start:07d}.npy"))
            start += SHARD_SIZE
        if start:
            print(f"  resuming from row {start:,} ({len(shards)} shards on disk)")

    t0 = time.time()
    for offset in range(start, n, SHARD_SIZE):
        block = texts[offset:offset + SHARD_SIZE]
        vecs = model.encode(
            block,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(out_dir / f"shard_{offset:07d}.npy", vecs)
        shards.append(vecs)

        done = min(offset + SHARD_SIZE, n)
        rate = (done - start) / max(time.time() - t0, 1e-9)
        eta_min = (n - done) / rate / 60 if rate else 0
        print(f"  {done:,}/{n:,}  {rate:.1f} docs/s  eta {eta_min:.0f} min")

    embeddings = np.vstack(shards) if shards else np.zeros((0, dim), np.float32)
    meta_path.write_text(json.dumps({
        "n": int(embeddings.shape[0]), "dim": dim,
        "normalized": normalize, "batch_size": batch_size,
    }, indent=2))
    return embeddings


def encode_queries(model, queries: list[str], instruction: str | None = None) -> np.ndarray:
    """Encode queries, applying a model's query-side instruction if it has one.

    BGE models are trained with an asymmetric prefix on the query side and
    lose measurable retrieval quality without it; MiniLM is symmetric and must
    not get one. Handling this per-model matters - applying the wrong
    convention would show up as a spurious embedding-model result in
    dimension 2.
    """
    texts = [f"{instruction}{q}" for q in queries] if instruction else queries
    return model.encode(
        texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


# Query-side instruction per model family. Empty string = symmetric model.
QUERY_INSTRUCTION = {
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
