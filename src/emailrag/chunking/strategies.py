"""The four chunking arms of dimension 1.

One reference tokenizer decides every boundary, for all strategies and all
embedding models. Chunking with each model's own tokenizer would make the
chunking and embedding dimensions interact, and the ablation is supposed to
vary them independently. All four candidate models are BERT-family WordPiece
with a 512-token limit, so a single reference tokenizer is faithful to what
each will actually see.

Chunk ids stay stable across strategies (``{strategy}:{dedup_key}:{ordinal}``)
so a retrieved chunk always maps back to exactly one source message. Metrics
are then computed at *message* level - if recall were measured over chunk ids,
a strategy that emits more chunks per message would score differently for a
reason that has nothing to do with retrieval quality, and the whole dimension
would be uninterpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

REFERENCE_TOKENIZER = "bert-base-uncased"
MAX_TOKENS = 512
OVERLAP_TOKENS = 64


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    dedup_key: str          # the source message - the unit metrics are scored on
    thread_id: str
    ordinal: int
    text: str
    n_tokens: int


def _header(msg: dict) -> str:
    """Prepended to every chunk: retrieval for "everything from Priya" is
    hopeless if the sender never appears in the embedded text."""
    date = msg.get("date_utc")
    date_s = date.strftime("%Y-%m-%d") if date is not None else "unknown-date"
    return (f"From: {msg.get('sender','')}\n"
            f"To: {(msg.get('recipients') or '').replace(';', ', ')}\n"
            f"Date: {date_s}\n"
            f"Subject: {msg.get('subject','')}\n\n")


class Chunker:
    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    # -- strategy 1 + 2 ----------------------------------------------------
    def fixed(self, msg: dict, overlap: int = 0) -> list[Chunk]:
        text = _header(msg) + (msg.get("body_new") or "")
        ids = self._encode(text)
        if not ids:
            return []
        stride = MAX_TOKENS - overlap
        out: list[Chunk] = []
        for ordinal, start in enumerate(range(0, len(ids), stride)):
            window = ids[start:start + MAX_TOKENS]
            if not window:
                break
            out.append(self._mk(msg, ordinal, self._decode(window), len(window)))
            if start + MAX_TOKENS >= len(ids):
                break
        return out

    # -- strategy 4 --------------------------------------------------------
    def whole_message(self, msg: dict) -> list[Chunk]:
        """No chunking at all. Most Enron messages are far under 512 tokens,
        so this arm tests whether chunking earns its complexity."""
        text = _header(msg) + (msg.get("body_new") or "")
        ids = self._encode(text)[:MAX_TOKENS]
        if not ids:
            return []
        return [self._mk(msg, 0, self._decode(ids), len(ids))]

    # -- strategy 3 --------------------------------------------------------
    def thread_aware(self, thread: list[dict]) -> list[Chunk]:
        """Pack whole messages from one thread into chunks, never splitting a
        message across a boundary. A message that alone exceeds the budget
        falls back to fixed splitting rather than being dropped."""
        out: list[Chunk] = []
        buf_ids: list[int] = []
        buf_owner: dict | None = None
        ordinal = 0

        def flush() -> None:
            nonlocal buf_ids, buf_owner, ordinal
            if buf_ids and buf_owner is not None:
                out.append(self._mk(buf_owner, ordinal, self._decode(buf_ids), len(buf_ids)))
                ordinal += 1
            buf_ids, buf_owner = [], None

        for msg in sorted(thread, key=lambda m: (m.get("date_utc") is None, m.get("date_utc"))):
            ids = self._encode(_header(msg) + (msg.get("body_new") or ""))
            if not ids:
                continue
            if len(ids) > MAX_TOKENS:
                flush()
                for c in self.fixed(msg, overlap=0):
                    out.append(c)
                continue
            if len(buf_ids) + len(ids) > MAX_TOKENS:
                flush()
            if buf_owner is None:
                buf_owner = msg
            buf_ids.extend(ids)
        flush()
        return out

    def _mk(self, msg: dict, ordinal: int, text: str, n_tokens: int) -> Chunk:
        key = msg["dedup_key"]
        return Chunk(
            chunk_id=f"{key}:{ordinal}",
            dedup_key=key,
            thread_id=msg.get("thread_id", key),
            ordinal=ordinal,
            text=text,
            n_tokens=n_tokens,
        )


# Registry consumed by the benchmark harness. `needs_thread` tells the caller
# whether to feed one message or a whole thread at a time.
STRATEGIES: dict[str, dict] = {
    "fixed_512":       {"needs_thread": False, "fn": lambda c, m: c.fixed(m, overlap=0)},
    "fixed_512_ov64":  {"needs_thread": False, "fn": lambda c, m: c.fixed(m, overlap=OVERLAP_TOKENS)},
    "whole_message":   {"needs_thread": False, "fn": lambda c, m: c.whole_message(m)},
    "thread_aware":    {"needs_thread": True,  "fn": lambda c, t: c.thread_aware(t)},
}


def chunk_corpus(strategy: str, tokenizer, messages: Iterable[dict]) -> list[Chunk]:
    spec = STRATEGIES[strategy]
    chunker = Chunker(tokenizer)
    fn: Callable = spec["fn"]
    messages = list(messages)

    if not spec["needs_thread"]:
        return [c for m in messages for c in fn(chunker, m)]

    threads: dict[str, list[dict]] = {}
    for m in messages:
        threads.setdefault(m.get("thread_id", m["dedup_key"]), []).append(m)
    # Sort thread keys so chunk emission order is deterministic run to run.
    return [c for tid in sorted(threads) for c in fn(chunker, threads[tid])]
