"""Recover the text behind a chunk id, without storing it twice.

Reranking needs chunk *text*: a cross-encoder scores (query, passage) pairs, so
the passages have to come from somewhere. Nothing in `data/index/<config>/`
holds them - the dense index is vectors, the BM25 index is a term matrix, and
`chunks.jsonl` deliberately stores only ids and metadata. Persisting the text
alongside would add roughly 120 MB per config, six times over, to hold a second
copy of a corpus already on disk.

So texts are rebuilt from `sample.parquet` on demand. Two things make that safe
rather than merely convenient:

*It is exact, not approximate.* Chunk ids are `{dedup_key}:{ordinal}`, and every
strategy assigns ordinals within one message (`fixed_*`, `whole_message`) or
within one thread (`thread_aware`). Selecting whole threads therefore reproduces
byte-identical chunks for the messages selected - the boundaries do not depend
on any message outside the thread. Selecting individual *messages* would not be
safe for `thread_aware`, which is why the unit of selection here is the thread.

*It is checked.* Every rebuild verifies the ids it produced against the ids the
index actually contains, and raises on any disagreement. A silent mismatch would
mean the reranker scoring text that does not correspond to the retrieved vector
- a plausible-looking dimension-4 result built on mislabelled passages.

That same check is what makes a GPU-built index trustworthy: `verify_parity`
proves a set of chunk ids produced elsewhere (a Kaggle notebook, say) describes
the same corpus as a local build.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from ..chunking import strategies as S

# A full rebuild is measured at 211 s for the 50k sample (58k chunks, 65 MB of
# text), which is fine once and irritating on every `make bench`. The rebuild is
# therefore cached beside the index it belongs to, gzipped. It lives under
# data/index/, which is already gitignored and already regenerable - this adds a
# derived artifact to a directory of derived artifacts, not a new source of
# truth. Delete it and the next run rebuilds it.
CACHE_NAME = "chunk_texts.jsonl.gz"


class ParityError(RuntimeError):
    """Rebuilt chunk ids disagree with the ids in the index."""


# Exactly what `_header` and the chunkers read, and nothing else.
CHUNKING_COLUMNS = ("dedup_key", "thread_id", "date_utc", "sender",
                    "recipients", "subject", "body_new")


def load_chunk_meta(path: Path) -> dict[str, dict]:
    """`chunks.jsonl` -> {chunk_id: {dedup_key, thread_id, n_tokens}}."""
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec
    return out


def verify_parity(local_ids: list[str], other_ids: list[str], *,
                  where: str = "index", ordered: bool = True) -> None:
    """Raise unless two chunk-id lists describe the same corpus.

    `ordered` matters for embeddings specifically: a dense index is a matrix
    whose row i belongs to chunk id i, so two builds that agree as *sets* but
    differ in order would silently pair every vector with the wrong chunk.
    """
    if len(local_ids) != len(other_ids):
        raise ParityError(
            f"{where}: {len(local_ids):,} local chunk ids vs {len(other_ids):,} - "
            f"different chunking, different corpus, or a partial build")

    if ordered:
        for i, (a, b) in enumerate(zip(local_ids, other_ids)):
            if a != b:
                raise ParityError(
                    f"{where}: chunk ids diverge at row {i}: {a!r} vs {b!r}. "
                    f"Row order is what pairs a vector with its chunk, so this "
                    f"cannot be reconciled by sorting.")
        return

    missing = set(local_ids) - set(other_ids)
    extra = set(other_ids) - set(local_ids)
    if missing or extra:
        raise ParityError(
            f"{where}: {len(missing)} id(s) only local (e.g. "
            f"{next(iter(missing), None)!r}), {len(extra)} only remote (e.g. "
            f"{next(iter(extra), None)!r})")


@dataclass
class ChunkTexts:
    """chunk_id -> text, for some subset of a config's chunks."""

    texts: dict[str, str]
    chunking: str
    n_messages_read: int = 0

    def __getitem__(self, chunk_id: str) -> str:
        return self.texts[chunk_id]

    def get(self, chunk_id: str, default: str = "") -> str:
        return self.texts.get(chunk_id, default)

    def __len__(self) -> int:
        return len(self.texts)


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(S.REFERENCE_TOKENIZER)


def read_messages(sample: Path) -> list[dict]:
    """The corpus, with only the columns the chunkers read."""
    return pq.read_table(sample, columns=list(CHUNKING_COLUMNS)).to_pylist()


def rebuild_chunks(sample: Path, chunking: str, tokenizer=None) -> list[S.Chunk]:
    """Full `Chunk` objects for the whole corpus, in emission order.

    Shared by the reranker's text lookup and by `import_embeddings.py`, which
    needs the ids *and* the texts: the ids to prove a GPU-built matrix describes
    this corpus, the texts to build the BM25 side locally.
    """
    if chunking not in S.STRATEGIES:
        raise ValueError(f"unknown chunking {chunking!r}; have {sorted(S.STRATEGIES)}")
    return S.chunk_corpus(chunking, tokenizer or _tokenizer(), read_messages(sample))


def rebuild(sample: Path, chunking: str, chunk_ids: list[str] | None = None,
            chunk_meta: dict[str, dict] | None = None,
            tokenizer=None) -> ChunkTexts:
    """Rebuild chunk texts for `chunk_ids` (or the whole corpus if None).

    `chunk_meta` is the config's `chunks.jsonl`. It is what maps a wanted chunk
    id to the thread that has to be re-chunked to produce it; without it the
    whole corpus is rebuilt, which is correct but slow.
    """
    if chunking not in S.STRATEGIES:
        raise ValueError(f"unknown chunking {chunking!r}; have {sorted(S.STRATEGIES)}")

    wanted = set(chunk_ids) if chunk_ids is not None else None
    messages = read_messages(sample)

    if wanted is not None and chunk_meta:
        # Whole threads, never individual messages: thread_aware packs several
        # messages into one chunk, so dropping a thread member would shift every
        # boundary after it.
        threads = {chunk_meta[c]["thread_id"] for c in wanted if c in chunk_meta}
        keys = {chunk_meta[c]["dedup_key"] for c in wanted if c in chunk_meta}
        messages = [m for m in messages
                    if m.get("thread_id") in threads or m.get("dedup_key") in keys]

    tokenizer = tokenizer or _tokenizer()
    chunks = S.chunk_corpus(chunking, tokenizer, messages)

    texts = {c.chunk_id: c.text for c in chunks
             if wanted is None or c.chunk_id in wanted}

    if wanted is not None:
        missing = wanted - set(texts)
        if missing:
            raise ParityError(
                f"{len(missing)} chunk id(s) in the index could not be rebuilt "
                f"from {sample} with chunking={chunking!r}, e.g. "
                f"{sorted(missing)[:3]}. The index and the sample are out of "
                f"sync - rebuild the index or repoint --sample.")

    return ChunkTexts(texts=texts, chunking=chunking, n_messages_read=len(messages))


def save_cache(path: Path, texts: ChunkTexts) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for chunk_id, text in texts.texts.items():
            fh.write(json.dumps({"chunk_id": chunk_id, "text": text},
                                ensure_ascii=False) + "\n")


def load_cache(path: Path, chunking: str) -> ChunkTexts:
    out: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["chunk_id"]] = rec["text"]
    return ChunkTexts(texts=out, chunking=chunking)


def load_or_build(index_dir: Path, sample: Path, chunking: str,
                  verbose: bool = True) -> ChunkTexts:
    """Chunk texts for a whole built config, from cache when possible.

    The cache is validated on load against the config's own `chunks.jsonl`, not
    trusted. A stale cache - left behind by a rebuilt index, or copied from
    another config - would feed the reranker passages belonging to different
    chunks, which is the one failure mode here that produces plausible numbers
    instead of an error.
    """
    index_dir, sample = Path(index_dir), Path(sample)
    cache = index_dir / CACHE_NAME
    meta = load_chunk_meta(index_dir / "chunks.jsonl")

    if cache.exists():
        texts = load_cache(cache, chunking)
        verify_parity(list(meta), list(texts.texts), where=str(cache), ordered=False)
        if verbose:
            print(f"  chunk texts: {len(texts):,} from {cache.name}")
        return texts

    if verbose:
        print(f"  chunk texts: rebuilding {len(meta):,} chunks from {sample.name} "
              f"(~3-4 min for the 50k sample; cached afterwards)")
    texts = rebuild(sample, chunking)
    verify_parity(list(meta), list(texts.texts), where="rebuild", ordered=False)
    save_cache(cache, texts)
    if verbose:
        print(f"  chunk texts: wrote {cache}")
    return texts
