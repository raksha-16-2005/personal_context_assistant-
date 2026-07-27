"""Exact dense retrieval over an in-memory matrix.

The ablations use exact search, not HNSW, on purpose. An approximate index
introduces its own recall loss, and that loss varies with dimensionality and
with how clustered a model's embedding space is - so an HNSW-backed comparison
of bge-base against MiniLM would be measuring the interaction of the model
*and* the index, and reporting it as a property of the model. Dimension 2 has
to isolate the model, so search here is exhaustive and the numbers are the
model's true ceiling.

pgvector/HNSW is used for the served system, where latency matters, and the
recall lost to approximation is measured there against these exact results -
which makes it a reportable number rather than a hidden confound.

At the 50k-message ablation scale a brute-force pass is a single BLAS call
over a ~150 MB matrix: a few milliseconds, faster than the ANN index it
replaces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, chunk_ids: list[str], matrix: np.ndarray) -> None:
        if len(chunk_ids) != matrix.shape[0]:
            raise ValueError(f"{len(chunk_ids)} ids vs {matrix.shape[0]} vectors")
        self.chunk_ids = np.asarray(chunk_ids, dtype=object)
        # float32 C-contiguous keeps the matmul in BLAS' fast path.
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def search(self, query_vec: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        """Cosine similarity. Vectors are L2-normalized at encode time, so the
        dot product *is* the cosine and no per-query renormalization is needed."""
        if self.matrix.shape[0] == 0:
            return []
        scores = self.matrix @ np.asarray(query_vec, dtype=np.float32).ravel()
        k = min(top_k, scores.shape[0])
        # argpartition is O(n) to find the top-k set; only that slice is sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(str(self.chunk_ids[i]), float(scores[i])) for i in top]

    def search_batch(self, query_matrix: np.ndarray, top_k: int = 100) -> list[list[tuple[str, float]]]:
        """One GEMM for all queries - substantially faster than looping when
        scoring all 80 eval queries against a config."""
        if self.matrix.shape[0] == 0:
            return [[] for _ in range(query_matrix.shape[0])]
        scores = np.asarray(query_matrix, dtype=np.float32) @ self.matrix.T
        k = min(top_k, scores.shape[1])
        out: list[list[tuple[str, float]]] = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top], kind="stable")]
            out.append([(str(self.chunk_ids[i]), float(row[i])) for i in top])
        return out

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.matrix)
        (path / "chunk_ids.txt").write_text("\n".join(self.chunk_ids.tolist()))

    @classmethod
    def load(cls, path: Path | str) -> "DenseIndex":
        path = Path(path)
        matrix = np.load(path / "embeddings.npy")
        chunk_ids = (path / "chunk_ids.txt").read_text().splitlines()
        return cls(chunk_ids, matrix)
