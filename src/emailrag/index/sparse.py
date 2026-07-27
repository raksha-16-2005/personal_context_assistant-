"""BM25 via `bm25s`.

`rank_bm25` is the usual choice in tutorials and is unusable at this scale -
it scores queries in pure Python against every document. `bm25s` builds a
scipy sparse matrix once and scores with vectorized ops, which is the
difference between a benchmark that finishes and one that does not.

Postgres full-text search was the other candidate, since the database is
already here. It is left out because `ts_rank_cd` is not BM25, and the point
of dimension 3 is to compare BM25 against dense retrieval on equal footing.
"""
from __future__ import annotations

from pathlib import Path

import bm25s
import Stemmer


class SparseIndex:
    def __init__(self, chunk_ids: list[str], retriever, stemmer) -> None:
        self.chunk_ids = chunk_ids
        self.retriever = retriever
        self.stemmer = stemmer

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "SparseIndex":
        stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(chunk_ids, retriever, stemmer)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer, show_progress=False)
        # bm25s raises if k exceeds the corpus size; clamp for small samples.
        k = min(top_k, len(self.chunk_ids))
        if k == 0:
            return []
        idxs, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        return [(self.chunk_ids[int(i)], float(s)) for i, s in zip(idxs[0], scores[0])]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path), corpus=None)
        (path / "chunk_ids.txt").write_text("\n".join(self.chunk_ids))

    @classmethod
    def load(cls, path: Path | str) -> "SparseIndex":
        path = Path(path)
        retriever = bm25s.BM25.load(str(path), load_corpus=False)
        chunk_ids = (path / "chunk_ids.txt").read_text().splitlines()
        return cls(chunk_ids, retriever, Stemmer.Stemmer("english"))
