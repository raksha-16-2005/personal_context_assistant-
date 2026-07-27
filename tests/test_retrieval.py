from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index.dense import DenseIndex
from emailrag.index.fusion import reciprocal_rank_fusion, weighted_score_fusion
from emailrag.index.sparse import SparseIndex
from emailrag.index.store import table_name


def _unit(rows: list[list[float]]) -> np.ndarray:
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# --- dense -----------------------------------------------------------------

def test_dense_ranks_by_cosine():
    idx = DenseIndex(["a:0", "b:0", "c:0"], _unit([[1, 0], [0.7, 0.7], [0, 1]]))
    got = idx.search(np.array([1.0, 0.0], np.float32), top_k=3)

    assert [d for d, _ in got] == ["a:0", "b:0", "c:0"]
    assert got[0][1] == pytest.approx(1.0, abs=1e-6)


def test_dense_batch_matches_single_query():
    # Ranking must be identical; scores only to float32 tolerance. The single
    # -query path is a GEMV and the batch path a GEMM, and BLAS accumulates
    # those in different orders - a ~1e-7 disagreement is expected and does
    # not affect ordering.
    rng = np.random.default_rng(0)
    mat = _unit(rng.normal(size=(200, 16)).tolist())
    idx = DenseIndex([f"m{i}:0" for i in range(200)], mat)
    queries = _unit(rng.normal(size=(5, 16)).tolist())

    batch = idx.search_batch(queries, top_k=10)
    for q, expected in zip(queries, batch):
        single = idx.search(q, top_k=10)
        assert [d for d, _ in single] == [d for d, _ in expected]
        np.testing.assert_allclose([s for _, s in single],
                                   [s for _, s in expected], rtol=1e-5, atol=1e-6)


def test_dense_top_k_clamps_to_corpus_size():
    idx = DenseIndex(["a:0"], _unit([[1, 0]]))
    assert len(idx.search(np.array([1.0, 0.0], np.float32), top_k=50)) == 1


def test_dense_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        DenseIndex(["a:0"], np.zeros((2, 4), np.float32))


def test_dense_roundtrips_through_disk(tmp_path):
    idx = DenseIndex(["a:0", "b:1"], _unit([[1, 0], [0, 1]]))
    idx.save(tmp_path / "d")
    back = DenseIndex.load(tmp_path / "d")

    assert back.chunk_ids.tolist() == ["a:0", "b:1"]
    np.testing.assert_allclose(back.matrix, idx.matrix)


# --- sparse ----------------------------------------------------------------

def test_bm25_finds_the_lexically_matching_chunk():
    texts = [
        "the quarterly audit report for the houston office",
        "lunch plans for friday afternoon",
        "pricing model discount tier for new contracts",
    ]
    idx = SparseIndex.build(["a:0", "b:0", "c:0"], texts)
    top = idx.search("discount tier pricing", top_k=3)

    assert top[0][0] == "c:0"


def test_bm25_top_k_clamps_to_corpus_size():
    idx = SparseIndex.build(["a:0"], ["only one document here"])
    assert len(idx.search("document", top_k=25)) == 1


def test_bm25_roundtrips_through_disk(tmp_path):
    idx = SparseIndex.build(["a:0", "b:0"], ["audit report houston", "friday lunch"])
    idx.save(tmp_path / "bm25")
    back = SparseIndex.load(tmp_path / "bm25")

    assert back.search("audit", top_k=1)[0][0] == "a:0"


# --- fusion ----------------------------------------------------------------

def test_rrf_rewards_agreement_across_runs():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    sparse = [("c", 40.0), ("a", 12.0), ("z", 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    # 'a' is 1st and 2nd; 'c' is 3rd and 1st. Agreement near the top wins.
    assert fused[0][0] == "a"
    assert {d for d, _ in fused} == {"a", "b", "c", "z"}


def test_rrf_ignores_score_magnitude():
    # BM25's unbounded scores must not dominate cosine's [-1,1].
    dense = [("a", 0.99), ("b", 0.98)]
    sparse_small = [("b", 0.2), ("a", 0.1)]
    sparse_huge = [("b", 20000.0), ("a", 10000.0)]

    assert reciprocal_rank_fusion([dense, sparse_small]) == \
           reciprocal_rank_fusion([dense, sparse_huge])


def test_rrf_is_deterministic_on_ties():
    run = [("b", 1.0), ("a", 1.0)]
    fused = reciprocal_rank_fusion([run, [("a", 1.0), ("b", 1.0)]])
    assert [d for d, _ in fused] == ["a", "b"]   # tie broken on doc_id


def test_weighted_fusion_respects_weights():
    dense = [("a", 1.0), ("b", 0.0)]
    sparse = [("b", 1.0), ("a", 0.0)]

    assert weighted_score_fusion([dense, sparse], [0.9, 0.1])[0][0] == "a"
    assert weighted_score_fusion([dense, sparse], [0.1, 0.9])[0][0] == "b"


def test_weighted_fusion_handles_a_degenerate_run():
    # All-equal scores must not divide by zero.
    fused = weighted_score_fusion([[("a", 5.0), ("b", 5.0)], [("a", 1.0)]], [0.5, 0.5])
    assert dict(fused)["a"] > dict(fused)["b"]


def test_weighted_fusion_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        weighted_score_fusion([[("a", 1.0)]], [0.5, 0.5])


def test_missing_doc_scores_zero_for_that_retriever():
    fused = dict(weighted_score_fusion([[("a", 1.0)], [("b", 1.0)]], [1.0, 1.0]))
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(1.0)


# --- store naming ----------------------------------------------------------

def test_table_names_are_deterministic_and_sql_safe():
    assert table_name("thread_aware", "BAAI/bge-base-en-v1.5") == \
           "chunks_thread_aware__bge_base_en_v1_5"
    assert table_name("fixed_512", "sentence-transformers/all-MiniLM-L6-v2") == \
           "chunks_fixed_512__all_minilm_l6_v2"


def test_table_name_rejects_injection():
    with pytest.raises(ValueError):
        table_name("fixed; DROP TABLE users--", "BAAI/bge-base-en-v1.5")
