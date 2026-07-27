from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.chunking import strategies as S
from emailrag.evaluation.metrics import (
    QueryResult, aggregate, by_query_class, collapse_to_messages,
    ndcg_at_k, recall_at_k, reciprocal_rank,
)

UTC = timezone.utc


class FakeTokenizer:
    """Whitespace tokenizer standing in for WordPiece.

    Chunk-boundary logic is what these tests exercise; using the real
    tokenizer here would make them slow and would test HuggingFace, not us.
    """

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 30000 for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _msg(key: str, body: str, thread: str = "t1", day: int = 1) -> dict:
    return {
        "dedup_key": key, "thread_id": thread, "sender": "a@enron.com",
        "recipients": "b@enron.com", "subject": "pricing", "body_new": body,
        "date_utc": datetime(2000, 12, day, tzinfo=UTC),
    }


@pytest.fixture
def tok():
    return FakeTokenizer()


def test_whole_message_emits_exactly_one_chunk(tok):
    chunks = S.chunk_corpus("whole_message", tok, [_msg("m1", "word " * 50)])
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "m1:0"
    assert chunks[0].dedup_key == "m1"


def test_whole_message_truncates_rather_than_splitting(tok):
    chunks = S.chunk_corpus("whole_message", tok, [_msg("m1", "word " * 2000)])
    assert len(chunks) == 1
    assert chunks[0].n_tokens == S.MAX_TOKENS


def test_fixed_splits_long_messages_and_numbers_ordinals(tok):
    chunks = S.chunk_corpus("fixed_512", tok, [_msg("m1", "word " * 1200)])
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.dedup_key == "m1" for c in chunks)
    assert all(c.n_tokens <= S.MAX_TOKENS for c in chunks)


def test_overlap_produces_more_chunks_than_no_overlap(tok):
    msg = _msg("m1", "word " * 2000)
    plain = S.chunk_corpus("fixed_512", tok, [msg])
    overlapped = S.chunk_corpus("fixed_512_ov64", tok, [msg])
    assert len(overlapped) > len(plain)


def test_thread_aware_never_splits_a_message_across_chunks(tok):
    # Three short messages in one thread pack into a single chunk.
    thread = [_msg(f"m{i}", "word " * 40, thread="t1", day=i + 1) for i in range(3)]
    chunks = S.chunk_corpus("thread_aware", tok, thread)
    assert len(chunks) == 1


def test_thread_aware_falls_back_to_splitting_an_oversized_message(tok):
    thread = [_msg("m1", "word " * 1500, thread="t1")]
    chunks = S.chunk_corpus("thread_aware", tok, thread)
    assert len(chunks) > 1
    assert all(c.n_tokens <= S.MAX_TOKENS for c in chunks)


def test_every_chunk_id_maps_back_to_one_message(tok):
    msgs = [_msg("m1", "word " * 900), _msg("m2", "word " * 30)]
    for name in S.STRATEGIES:
        chunks = S.chunk_corpus(name, tok, msgs)
        assert chunks, name
        for c in chunks:
            assert c.chunk_id.rsplit(":", 1)[0] == c.dedup_key, name


def test_header_is_embedded_so_sender_is_retrievable(tok):
    header = S._header(_msg("m1", "body"))
    assert "a@enron.com" in header and "pricing" in header and "2000-12-01" in header


def test_chunking_is_deterministic(tok):
    msgs = [_msg(f"m{i}", "word " * (i * 37 + 20), thread=f"t{i % 3}") for i in range(20)]
    for name in S.STRATEGIES:
        a = [c.chunk_id for c in S.chunk_corpus(name, tok, msgs)]
        b = [c.chunk_id for c in S.chunk_corpus(name, tok, msgs)]
        assert a == b, name


# --- metrics ---------------------------------------------------------------

def test_collapse_keeps_best_position_per_message():
    ranked = [("m3:0", .9), ("m1:2", .8), ("m3:1", .7), ("m1:0", .6), ("m2:0", .5)]
    assert collapse_to_messages(ranked) == ["m3", "m1", "m2"]


def test_ndcg_matches_hand_calculation():
    # rel at ranks 2 and 5: DCG = 1/log2(3) + 1/log2(6); IDCG = 1 + 1/log2(3)
    ranked, rel = ["a", "b", "c", "d", "e"], {"b", "e"}
    expected = (1 / math.log2(3) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 10) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_recall_and_mrr():
    ranked, rel = ["x", "b", "c"], {"b", "z"}
    assert recall_at_k(ranked, rel, 3) == 0.5
    assert reciprocal_rank(ranked, rel) == 0.5
    assert reciprocal_rank(["x", "y"], rel) == 0.0


def test_unanswerable_queries_are_undefined_not_zero():
    assert math.isnan(recall_at_k(["a"], set(), 5))
    assert math.isnan(ndcg_at_k(["a"], set(), 10))


def _qr(qid, cls, ranked, rel, ms=10.0):
    return QueryResult(qid, cls, ranked, rel, ms)


def test_aggregate_excludes_unanswerable_controls():
    results = [
        _qr("q1", "semantic", ["a", "b"], {"a"}),
        _qr("q2", "unanswerable", ["c", "d"], set()),
    ]
    agg = aggregate(results, ks=(5,))

    assert agg["n_queries"] == 2
    assert agg["n_answerable"] == 1
    # q2 must not drag the mean toward zero.
    assert agg["recall@5"] == 1.0
    assert agg["ndcg@10"] == pytest.approx(1.0)


def test_aggregate_reports_latency_over_all_queries():
    results = [_qr(f"q{i}", "semantic", ["a"], {"a"}, ms=float(i)) for i in range(1, 101)]
    agg = aggregate(results)
    assert agg["p50_ms"] == 51.0
    assert agg["p95_ms"] == 96.0


def test_per_class_breakdown_separates_the_router_signal():
    results = [
        _qr("q1", "semantic", ["a", "b"], {"a"}),
        _qr("q2", "temporal", ["x", "y"], {"z"}),   # dense misses entirely
    ]
    per_class = by_query_class(results, ks=(5,))

    assert per_class["semantic"]["recall@5"] == 1.0
    assert per_class["temporal"]["recall@5"] == 0.0
