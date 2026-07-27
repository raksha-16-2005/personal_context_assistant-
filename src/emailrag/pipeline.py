"""One question in, a cited answer out - the assembled system.

The ablation harness (`evaluation/harness.py`) and this module solve different
problems and are deliberately separate. The harness runs 80 queries through many
configurations and reports metrics; it holds no message metadata, because metrics
never need it. This runs *one* question through *one* configuration and has to
produce something a person can read: sender, date, subject, the excerpt, and a
link back to the message. Sharing one code path would mean either loading message
metadata for 80x6x4 benchmark rows that ignore it, or shipping a UI that can only
display chunk ids.

What they do share is every component - the same index, the same fusion, the same
reranker, the same transforms - so a configuration that wins the ablation is the
configuration this serves. The `RunConfig` names are the same strings.

Loading is lazy and cached per process: the dense matrix is ~90 MB, the chunk
texts ~65 MB, and a UI that reloaded them per request would be unusable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .generation.synthesize import Answer, Citation, Synthesizer
from .index import chunktext as CT
from .index import embed as E
from .index import rerank as RR
from .index.dense import DenseIndex
from .index.fusion import reciprocal_rank_fusion, weighted_score_fusion
from .index.sparse import SparseIndex
from .query.transform import QueryTransformer

# Chunks pulled before collapsing to messages. Deeper than the number of sources
# shown, for the same reason the harness retrieves deep: several chunks routinely
# belong to one message, so 60 chunks can be far fewer than 60 messages.
RETRIEVE_DEPTH = 120

DEFAULT_RERANK = "L2@20/t192"      # the only arm inside the 200 ms CPU budget


@dataclass(slots=True)
class RetrievedMessage:
    message_id: str
    score: float
    rank: int
    chunk_ids: list[str] = field(default_factory=list)
    text: str = ""
    sender: str = ""
    recipients: str = ""
    date: str = ""
    subject: str = ""


@dataclass(slots=True)
class SearchResult:
    question: str
    messages: list[RetrievedMessage]
    timings: dict = field(default_factory=dict)
    transform_kind: str = "none"
    transform_texts: list[str] = field(default_factory=list)
    transform_degraded: bool = False


def collapse(ranked: list[tuple[str, float]], top_n: int,
             texts: dict[str, str], headers: dict[str, dict]
             ) -> list[RetrievedMessage]:
    """Chunk ranking -> message ranking, keeping each message's best position.

    A module-level function rather than a method because it is the part of this
    file with rules worth testing, and it needs nothing from the 155 MB of loaded
    index to do its job.

    A message's several retrieved chunks are concatenated in rank order, so a
    citation shows everything retrieval found in that message rather than an
    arbitrary one of its fragments. `top_n` counts *messages*, not chunks - the
    caller asked for ten things to read, not ten fragments of three things.
    """
    order: list[str] = []
    by_message: dict[str, RetrievedMessage] = {}

    for chunk_id, score in ranked:
        key = chunk_id.rsplit(":", 1)[0]
        if key not in by_message:
            if len(by_message) >= top_n:
                # Full on messages, but later chunks of messages already in the
                # list still get appended below - dropping them would show a
                # partial excerpt of a message we retrieved twice.
                continue
            head = headers.get(key, {})
            by_message[key] = RetrievedMessage(
                message_id=key, score=float(score), rank=len(order) + 1,
                sender=head.get("sender", ""),
                recipients=head.get("recipients", ""),
                date=head.get("date", ""),
                subject=head.get("subject", ""),
            )
            order.append(key)
        msg = by_message[key]
        msg.chunk_ids.append(chunk_id)
        piece = texts.get(chunk_id, "")
        if piece and piece not in msg.text:
            msg.text = f"{msg.text}\n{piece}".strip() if msg.text else piece

    return [by_message[k] for k in order]


class Pipeline:
    """A loaded index plus the optional reranker and transform, ready to query."""

    def __init__(self, index_dir: Path, sample: Path, *, retriever: str = "hybrid_rrf",
                 rerank: str = DEFAULT_RERANK, transform: str = "none",
                 dense_weight: float = 0.5, threads: int = 6,
                 verbose: bool = True) -> None:
        self.index_dir = Path(index_dir)
        self.sample = Path(sample)
        self.retriever = retriever
        self.rerank = rerank
        self.transform = transform
        self.dense_weight = dense_weight
        self.verbose = verbose

        meta_path = self.index_dir / "config.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no built index at {self.index_dir}.\n"
                f"  build one: make index-pilot")
        self.meta = json.loads(meta_path.read_text())
        self.chunking = self.meta["chunking"]
        self.model_id = self.meta["model"]

        E.configure_threads(threads)
        self._log(f"loading {self.index_dir.name} ...")
        self.bm25 = SparseIndex.load(self.index_dir / "bm25")
        self.dense = DenseIndex.load(self.index_dir / "dense")
        self.model = E.load_model(self.model_id)
        self.instruction = E.QUERY_INSTRUCTION.get(self.model_id, "")

        self.chunk_meta = CT.load_chunk_meta(self.index_dir / "chunks.jsonl")
        self.texts = CT.load_or_build(self.index_dir, self.sample, self.chunking,
                                      verbose=verbose)
        self._messages = self._load_message_metadata()

        self.reranker = None
        spec = RR.SPECS.get(rerank)
        if spec is not None:
            self._log(f"reranker: {spec.label}")
            self.reranker = RR.CrossEncoderReranker.from_spec(spec)
        self.rerank_top_k = spec.top_k if spec else 0

        self.transformer = (QueryTransformer(transform) if transform != "none"
                            else None)
        self.synthesizer = Synthesizer()

        # First encode allocates buffers and materialises lazily-loaded weights -
        # ~1.3 s measured. Paying it at load time keeps the first user query from
        # looking 40x slower than every later one.
        E.encode_queries(self.model, ["warm up"], self.instruction or None)
        self.bm25.search("warm up", top_k=1)
        self._log(f"ready: {len(self.chunk_meta):,} chunks, "
                  f"{len(self._messages):,} messages")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}", flush=True)

    def _load_message_metadata(self) -> dict[str, dict]:
        """dedup_key -> the header fields a citation needs.

        Only headers, never bodies: the body a citation should show is the chunk
        that was retrieved, which is already in `self.texts`. Loading bodies here
        would double the memory for text nobody displays.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(self.sample, columns=[
            "dedup_key", "sender", "recipients", "subject", "date_utc"])
        out = {}
        for row in table.to_pylist():
            date = row.get("date_utc")
            out[row["dedup_key"]] = {
                "sender": row.get("sender") or "",
                "recipients": row.get("recipients") or "",
                "subject": row.get("subject") or "",
                "date": date.strftime("%Y-%m-%d") if date is not None else "",
            }
        return out

    # -- retrieval ---------------------------------------------------------

    def _rank(self, texts_dense: list[str], texts_sparse: list[str]
              ) -> list[tuple[str, float]]:
        dense_run: list[tuple[str, float]] = []
        sparse_run: list[tuple[str, float]] = []

        if self.retriever != "bm25":
            qvecs = E.encode_queries(self.model, texts_dense, self.instruction or None)
            runs = [self.dense.search(v, top_k=RETRIEVE_DEPTH) for v in qvecs]
            dense_run = (runs[0] if len(runs) == 1
                         else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))
        if self.retriever != "dense":
            runs = [self.bm25.search(t, top_k=RETRIEVE_DEPTH) for t in texts_sparse]
            sparse_run = (runs[0] if len(runs) == 1
                          else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))

        if self.retriever == "bm25":
            return sparse_run
        if self.retriever == "dense":
            return dense_run
        if self.retriever == "hybrid_rrf":
            return reciprocal_rank_fusion([dense_run, sparse_run], top_k=RETRIEVE_DEPTH)
        if self.retriever == "hybrid_weighted":
            return weighted_score_fusion(
                [dense_run, sparse_run],
                [self.dense_weight, 1.0 - self.dense_weight], top_k=RETRIEVE_DEPTH)
        raise ValueError(f"unknown retriever {self.retriever!r}")

    def search(self, question: str, top_n: int = 10, as_of: str = "") -> SearchResult:
        """Retrieve, rerank, and collapse chunks to messages with metadata."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        if self.transformer is not None:
            tq = self.transformer(question, as_of=as_of)
        else:
            from .query.transform import identity
            tq = identity(question)
        timings["transform_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked = self._rank(tq.dense_texts, tq.sparse_texts)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        timings["rerank_ms"] = 0.0
        if self.reranker is not None:
            # Reranked against the user's actual question, never the transform's
            # output: a HyDE document is a fabricated email, and scoring passages
            # for similarity to a fabrication measures agreement with the
            # generator rather than relevance to the question.
            ranked, ms = self.reranker.rerank_timed(
                question, ranked, self.texts.texts, self.rerank_top_k)
            timings["rerank_ms"] = ms

        messages = self._collapse(ranked, top_n)
        timings["total_ms"] = sum(timings.values())

        return SearchResult(
            question=question, messages=messages, timings=timings,
            transform_kind=tq.kind,
            transform_texts=[t for t in tq.dense_texts if t != question],
            transform_degraded=tq.degraded,
        )

    def _collapse(self, ranked: list[tuple[str, float]], top_n: int
                  ) -> list[RetrievedMessage]:
        return collapse(ranked, top_n, self.texts.texts, self._messages)

    # -- generation --------------------------------------------------------

    def ask(self, question: str, n_sources: int = 6, as_of: str = "",
            top_n: int = 10) -> tuple[Answer, SearchResult]:
        """The whole thing: retrieve, rerank, synthesize a cited answer."""
        result = self.search(question, top_n=max(top_n, n_sources), as_of=as_of)
        citations = [
            Citation(n=i + 1, message_id=m.message_id, sender=m.sender,
                     recipients=m.recipients, date=m.date, subject=m.subject,
                     text=m.text, score=m.score, chunk_ids=list(m.chunk_ids))
            for i, m in enumerate(result.messages[:n_sources])
        ]
        self.synthesizer.n_sources = n_sources
        answer = self.synthesizer.answer(question, citations, as_of=as_of)
        return answer, result

    @property
    def config_summary(self) -> dict:
        return {
            "index": self.index_dir.name,
            "chunking": self.chunking,
            "model": self.model_id,
            "retriever": self.retriever,
            "rerank": self.rerank,
            "transform": self.transform,
            "n_chunks": len(self.chunk_meta),
            "n_messages": len(self._messages),
            "embed_platform": self.meta.get("embed_platform", "local-cpu"),
        }
