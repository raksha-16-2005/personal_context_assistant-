"""Dimension 4: cross-encoder reranking over the retrieved head.

A bi-encoder scores a query and a passage in separate forward passes, so the
passage embedding cannot depend on the query. A cross-encoder concatenates them
into one sequence and attends across both, which is strictly more expressive and
strictly more expensive - the cost is per (query, passage) pair, not per corpus.
That trade is the entire dimension: does the reordering buy enough nDCG to pay
for the latency, and at what k.

Measured on this machine (i7-9750H, CPU, see docs/HARDWARE.md):

    cross-encoder/ms-marco-MiniLM-L-6-v2   962 ms @ top-50
    BAAI/bge-reranker-base               6,802 ms @ top-50

6.8 s/query is not a system anyone would ship, which is why `bge-reranker-base`
is reported as an offline quality ceiling rather than a candidate, and why the
shipped path is MiniLM over top-20 with an int8 ONNX export.

**Rerank the head, keep the tail.** `rerank` reorders the top-k chunks and
appends everything below them in their original order rather than truncating.
This is not tidiness - metrics here are message-level over a deep chunk ranking
(see evaluation/harness.py). Truncating at k would cut the number of *distinct
messages* available to recall@20, so a reranker that reordered the head
perfectly could still show a recall drop caused entirely by truncation. Keeping
the tail makes dimension 4 measure reordering, and only reordering.

**Passages are the text the index holds, not the original message.** Chunking
round-trips text through the reference tokenizer, so an indexed chunk is
lowercased and detokenized ("stacey. richardson @ enron. com"). Scoring the
pristine original instead would score a passage the retriever never saw. Both
models here are uncased BERT-family, so the cost is punctuation spacing; the
consistency is worth more than the spacing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

# Where `scripts/export_onnx_reranker.py` writes quantized models.
ONNX_ROOT = Path(__file__).resolve().parents[3] / "data" / "onnx"

MAX_LENGTH = 512


@dataclass(frozen=True, slots=True)
class RerankSpec:
    """One dimension-4 arm: which model, how deep, and how much of each chunk.

    `max_length` is a first-class arm parameter, not a detail, because it is the
    cheapest latency lever there is. Attention cost grows superlinearly in
    sequence length, and `make bench-rerank-budget` measures 128-token passages
    at 2-5x the throughput of 512-token ones on this CPU. Unlike lowering
    `top_k`, truncation does not shrink the candidate set - every chunk still
    gets scored, just on less of itself. Whether the discarded tail mattered is
    exactly what the dimension-4 table is for.
    """

    model_id: str
    top_k: int
    max_length: int = MAX_LENGTH
    onnx_int8: bool = False

    @property
    def label(self) -> str:
        short = self.model_id.split("/")[-1].replace("ms-marco-MiniLM-", "")
        suffix = "+onnx-int8" if self.onnx_int8 else ""
        length = f"/t{self.max_length}" if self.max_length != MAX_LENGTH else ""
        return f"{short}@{self.top_k}{length}{suffix}"


# The arms actually reported. `none` is the baseline row - dimension 4 is
# meaningless without the un-reranked ranking beside it.
#
# Latency figures are the medians from `make bench-rerank-budget` on this CPU
# (runs/rerank_budget.json), at the pilot index's mean chunk length of ~300
# tokens. They are the reason for the arm selection, and they are not portable
# to other hardware.
SPECS: dict[str, RerankSpec | None] = {
    "none": None,
    # Quality reference: the deepest, longest, most expensive thing that still
    # finishes. 2.7 s/query - an offline ceiling, not a candidate.
    "L6@50": RerankSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", 50),
    # The plan's original shipped config. Measured at 628 ms: 3x over budget,
    # which is the finding that made the arms below necessary.
    "L6@20": RerankSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", 20),
    # Same model, truncated passages. Isolates "does the tail of a chunk matter"
    # from "does a smaller model matter".
    "L6@20/t192": RerankSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", 20, max_length=192),
    # The shipped candidate: 158 ms, inside the 200 ms budget. Two levers at
    # once (2 layers instead of 6, 192 tokens instead of 512), so it needs the
    # two arms above beside it to attribute whatever quality it loses.
    "L2@20/t192": RerankSpec("cross-encoder/ms-marco-MiniLM-L-2-v2", 20, max_length=192),
    # Depth instead of length, at a comparable price (128 ms).
    "L2@10": RerankSpec("cross-encoder/ms-marco-MiniLM-L-2-v2", 10, max_length=300),
    # int8 export of the plan's config. Kept because the *negative* result is
    # worth publishing: on this CPU (AVX2, no VNNI) it is slower than torch fp32
    # at realistic passage length - see scripts/export_onnx_reranker.py.
    "L6@20-onnx": RerankSpec("cross-encoder/ms-marco-MiniLM-L-6-v2", 20, onnx_int8=True),
    # The quality ceiling from the plan. 6.8 s/query; reported, never shipped.
    "bge@50": RerankSpec("BAAI/bge-reranker-base", 50),
}

# What `make bench --dimension rerank` runs without an explicit --rerank list:
# the baseline, the two arms that isolate each lever, and the shipped candidate.
# The 2.7 s and 6.8 s arms are opt-in - they cost hours over a full eval set.
DEFAULT_ARMS = ("none", "L6@20", "L6@20/t192", "L2@20/t192")


def onnx_dir(model_id: str) -> Path:
    return ONNX_ROOT / model_id.replace("/", "__")


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a torch or ONNX cross-encoder."""

    def __init__(self, model, tokenizer, *, model_id: str, batch_size: int = 16,
                 max_length: int = MAX_LENGTH, backend: str = "torch") -> None:
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.backend = backend

    @classmethod
    def from_spec(cls, spec: RerankSpec, **kw) -> "CrossEncoderReranker":
        return cls.load(spec.model_id, onnx_int8=spec.onnx_int8,
                        max_length=spec.max_length, **kw)

    @classmethod
    def load(cls, model_id: str, *, onnx_int8: bool = False, batch_size: int = 16,
             max_length: int = MAX_LENGTH, threads: int = 6) -> "CrossEncoderReranker":
        from transformers import AutoTokenizer

        if onnx_int8:
            # Checked before anything is downloaded: a missing export should
            # print the one command that fixes it, not fail deep inside a hub
            # request for a model that was never exported.
            path = onnx_dir(model_id)
            if not (path / "model_quantized.onnx").exists():
                raise FileNotFoundError(
                    f"no int8 export at {path}.\n"
                    f"  build it: .venv/bin/python scripts/export_onnx_reranker.py "
                    f"--model {model_id}")
            from optimum.onnxruntime import ORTModelForSequenceClassification
            # The tokenizer comes from the export directory, not the hub: the
            # export saved the one it was traced with, and a mismatched
            # tokenizer is a silent correctness bug rather than a load error.
            model = ORTModelForSequenceClassification.from_pretrained(
                path, file_name="model_quantized.onnx")
            return cls(model, AutoTokenizer.from_pretrained(path), model_id=model_id,
                       batch_size=batch_size, max_length=max_length,
                       backend="onnx-int8")

        import torch
        from transformers import AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        torch.set_num_threads(threads)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        return cls(model, tokenizer, model_id=model_id, batch_size=batch_size,
                   max_length=max_length, backend="torch")

    # -- scoring -----------------------------------------------------------

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance score per passage, higher is better."""
        if not passages:
            return np.zeros(0, dtype=np.float32)

        import torch

        out: list[np.ndarray] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = self.tok([query] * len(batch), batch, padding=True,
                               truncation="longest_first", max_length=self.max_length,
                               return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            logits = logits.detach().cpu().numpy()
            # Rerankers are trained either as regressors (one logit) or as
            # binary classifiers (two). Taking column 0 of a 2-logit model would
            # rank by *irrelevance* - a silent, plausible-looking inversion.
            out.append(logits[:, 0] if logits.shape[1] == 1
                       else logits[:, 1] - logits[:, 0])
        return np.concatenate(out).astype(np.float32)

    def rerank(self, query: str, ranked: list[tuple[str, float]],
               texts: Mapping[str, str], top_k: int) -> list[tuple[str, float]]:
        """Reorder the top-k of `ranked`; leave everything below it untouched.

        Chunks whose text is unavailable are dropped from the head and appended
        to the front of the tail rather than scored against an empty string,
        which would rank them arbitrarily instead of leaving them where
        retrieval put them.
        """
        head, tail = ranked[:top_k], ranked[top_k:]
        scorable = [(cid, s) for cid, s in head if texts.get(cid)]
        unscorable = [(cid, s) for cid, s in head if not texts.get(cid)]

        if scorable:
            scores = self.score(query, [texts[cid] for cid, _ in scorable])
            order = np.argsort(-scores, kind="stable")
            reranked = [(scorable[i][0], float(scores[i])) for i in order]
        else:
            reranked = []

        # The tail keeps its retrieval order. Its scores are rewritten to sit
        # strictly below the reranked head so the combined list stays sorted by
        # score - anything consuming this as a ranked run (fusion, a
        # score-thresholded UI) would otherwise see a discontinuity.
        floor = min([s for _, s in reranked], default=0.0)
        below = unscorable + tail
        offset = floor - 1.0
        tail_scored = [(cid, offset - i) for i, (cid, _) in enumerate(below)]
        return reranked + tail_scored

    def rerank_timed(self, query: str, ranked: list[tuple[str, float]],
                     texts: Mapping[str, str], top_k: int
                     ) -> tuple[list[tuple[str, float]], float]:
        """`rerank` plus the milliseconds it took - the added latency that
        dimension 4's nDCG gain has to justify."""
        t0 = time.perf_counter()
        out = self.rerank(query, ranked, texts, top_k)
        return out, (time.perf_counter() - t0) * 1000
