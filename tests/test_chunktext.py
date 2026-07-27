from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index import chunktext as CT


class FakeTokenizer:
    """Whitespace tokenizer standing in for bert-base-uncased.

    The real one costs seconds to load and is what `test_corpus` already covers;
    what matters here is that chunk ids round-trip, not how WordPiece splits.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


MESSAGES = [
    {"dedup_key": "k1", "thread_id": "t1", "sender": "a@x.com",
     "recipients": "b@x.com", "subject": "pricing", "body_new": "first in thread",
     "date_utc": datetime(2001, 5, 1, tzinfo=timezone.utc)},
    {"dedup_key": "k2", "thread_id": "t1", "sender": "b@x.com",
     "recipients": "a@x.com", "subject": "re: pricing", "body_new": "second in thread",
     "date_utc": datetime(2001, 5, 2, tzinfo=timezone.utc)},
    {"dedup_key": "k3", "thread_id": "t2", "sender": "c@x.com",
     "recipients": "d@x.com", "subject": "audit", "body_new": "unrelated thread",
     "date_utc": datetime(2001, 5, 3, tzinfo=timezone.utc)},
]


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist(MESSAGES), path)
    return path


def _write_meta(path: Path, chunks) -> Path:
    with open(path, "w") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "dedup_key": c.dedup_key,
                                 "thread_id": c.thread_id, "n_tokens": c.n_tokens}) + "\n")
    return path


def _chunks(chunking):
    from emailrag.chunking import strategies as S
    return S.chunk_corpus(chunking, FakeTokenizer(), MESSAGES)


# -- parity -----------------------------------------------------------------

def test_parity_passes_on_identical_lists():
    CT.verify_parity(["a:0", "b:0"], ["a:0", "b:0"])


def test_parity_rejects_a_different_count():
    with pytest.raises(CT.ParityError, match="different chunking"):
        CT.verify_parity(["a:0", "b:0"], ["a:0"])


def test_parity_rejects_reordering_because_row_order_pairs_vectors_to_chunks():
    # A dense index is a matrix whose row i belongs to chunk id i. Two builds
    # agreeing as sets but differing in order would pair every vector with the
    # wrong chunk - the exact failure a GPU-built index could introduce.
    with pytest.raises(CT.ParityError, match="row 0"):
        CT.verify_parity(["a:0", "b:0"], ["b:0", "a:0"])


def test_parity_can_ignore_order_when_only_membership_matters():
    CT.verify_parity(["a:0", "b:0"], ["b:0", "a:0"], ordered=False)


def test_unordered_parity_still_catches_a_missing_id():
    with pytest.raises(CT.ParityError, match="only local"):
        CT.verify_parity(["a:0", "b:0"], ["a:0", "c:0"], ordered=False)


# -- rebuild ----------------------------------------------------------------

def test_full_rebuild_reproduces_every_chunk(sample, monkeypatch):
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    expected = _chunks("whole_message")

    out = CT.rebuild(sample, "whole_message")

    assert list(out.texts) == [c.chunk_id for c in expected]
    assert out[expected[0].chunk_id] == expected[0].text


def test_selective_rebuild_pulls_whole_threads_not_single_messages(sample, tmp_path,
                                                                  monkeypatch):
    # thread_aware packs several messages into one chunk, so dropping a thread
    # member would shift every boundary after it. Asking for a chunk owned by k1
    # must therefore re-read k2 as well.
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    chunks = _chunks("thread_aware")
    meta = CT.load_chunk_meta(_write_meta(tmp_path / "chunks.jsonl", chunks))
    target = [c.chunk_id for c in chunks if c.dedup_key in ("k1", "k2")][:1]

    out = CT.rebuild(sample, "thread_aware", target, meta)

    assert list(out.texts) == target
    assert out.n_messages_read == 2          # both of thread t1, not just one
    assert out[target[0]] == next(c.text for c in chunks if c.chunk_id == target[0])


def test_selective_rebuild_is_byte_identical_to_the_full_one(sample, tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    chunks = _chunks("fixed_512")
    meta = CT.load_chunk_meta(_write_meta(tmp_path / "chunks.jsonl", chunks))
    wanted = [chunks[-1].chunk_id]

    partial = CT.rebuild(sample, "fixed_512", wanted, meta)
    full = CT.rebuild(sample, "fixed_512")

    assert partial[wanted[0]] == full[wanted[0]]


def test_a_chunk_id_that_cannot_be_rebuilt_raises(sample, tmp_path, monkeypatch):
    # An index built from a different sample would otherwise silently feed the
    # reranker nothing for those chunks.
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    meta = {"ghost:0": {"dedup_key": "ghost", "thread_id": "t9", "n_tokens": 1}}

    with pytest.raises(CT.ParityError, match="out of\n?\\s*sync|could not be rebuilt"):
        CT.rebuild(sample, "whole_message", ["ghost:0"], meta)


def test_unknown_chunking_is_rejected(sample):
    with pytest.raises(ValueError, match="unknown chunking"):
        CT.rebuild(sample, "telepathy")


def test_only_the_columns_the_chunkers_read_are_loaded():
    # `body` (the quoted original) is the largest column in the sample and no
    # strategy uses it.
    assert "body" not in CT.CHUNKING_COLUMNS
    assert "body_new" in CT.CHUNKING_COLUMNS


# -- disk cache -------------------------------------------------------------

def test_cache_roundtrips(tmp_path, sample, monkeypatch):
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    built = CT.rebuild(sample, "whole_message")
    path = tmp_path / CT.CACHE_NAME

    CT.save_cache(path, built)
    loaded = CT.load_cache(path, "whole_message")

    assert loaded.texts == built.texts
    assert gzip.open(path, "rt").readline().startswith("{")


def test_load_or_build_writes_a_cache_then_reuses_it(tmp_path, sample, monkeypatch):
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    index_dir = tmp_path / "cfg"
    index_dir.mkdir()
    _write_meta(index_dir / "chunks.jsonl", _chunks("whole_message"))

    first = CT.load_or_build(index_dir, sample, "whole_message", verbose=False)
    assert (index_dir / CT.CACHE_NAME).exists()

    # Second call must not touch the corpus at all.
    monkeypatch.setattr(CT, "rebuild",
                        lambda *a, **k: pytest.fail("rebuilt despite a valid cache"))
    second = CT.load_or_build(index_dir, sample, "whole_message", verbose=False)

    assert second.texts == first.texts


def test_a_stale_cache_is_rejected_not_trusted(tmp_path, sample, monkeypatch):
    # A cache left behind by a rebuilt index would feed the reranker passages
    # belonging to different chunks - the one failure mode here that produces
    # plausible numbers instead of an error.
    monkeypatch.setattr(CT, "_tokenizer", FakeTokenizer)
    index_dir = tmp_path / "cfg"
    index_dir.mkdir()
    _write_meta(index_dir / "chunks.jsonl", _chunks("whole_message"))
    CT.save_cache(index_dir / CT.CACHE_NAME,
                  CT.ChunkTexts(texts={"someone-elses:0": "text"},
                                chunking="whole_message"))

    with pytest.raises(CT.ParityError):
        CT.load_or_build(index_dir, sample, "whole_message", verbose=False)
