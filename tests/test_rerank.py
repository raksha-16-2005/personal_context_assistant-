from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index import rerank as RR


class FakeCrossEncoder(RR.CrossEncoderReranker):
    """Scores by keyword overlap, so the expected ordering is known exactly.

    Loading a real cross-encoder would make this suite minutes long and would be
    testing HuggingFace rather than the head/tail contract, which is where the
    bugs that corrupt dimension 4 live.
    """

    def __init__(self, **kw):
        self.calls: list[list[str]] = []
        super().__init__(model=None, tokenizer=None, model_id="fake", **kw)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        self.calls.append(passages)
        terms = set(query.lower().split())
        return np.array([len(terms & set(p.lower().split())) for p in passages],
                        dtype=np.float32)


TEXTS = {
    "m1:0": "pricing model discount tier decision",
    "m2:0": "quarterly audit findings houston",
    "m3:0": "lunch plans friday",
    "m4:0": "discount tier pricing memo",
    "m5:0": "unrelated vendor invoice",
}
RANKED = [("m3:0", 9.0), ("m2:0", 8.0), ("m1:0", 7.0), ("m4:0", 6.0), ("m5:0", 5.0)]


def test_reranking_moves_the_relevant_chunk_to_the_top():
    rr = FakeCrossEncoder()
    out = rr.rerank("pricing discount tier", RANKED, TEXTS, top_k=5)

    assert [cid for cid, _ in out][:2] == ["m1:0", "m4:0"]


def test_the_tail_below_top_k_is_kept_in_its_original_order():
    # Truncating at k would shrink the number of distinct messages available to
    # recall@20, so a perfectly-reordered head could still show a recall drop
    # caused entirely by truncation.
    rr = FakeCrossEncoder()
    out = rr.rerank("pricing discount tier", RANKED, TEXTS, top_k=2)

    ids = [cid for cid, _ in out]
    assert len(ids) == len(RANKED)                 # nothing dropped
    assert ids[2:] == ["m1:0", "m4:0", "m5:0"]     # tail untouched, in order
    assert rr.calls == [[TEXTS["m3:0"], TEXTS["m2:0"]]]   # only the head scored


def test_scores_stay_monotonically_decreasing_across_the_boundary():
    rr = FakeCrossEncoder()
    out = rr.rerank("pricing discount tier", RANKED, TEXTS, top_k=3)

    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_chunks_with_no_text_are_left_where_retrieval_put_them():
    # Scoring a missing chunk against an empty string would rank it arbitrarily.
    rr = FakeCrossEncoder()
    ranked = [("ghost:0", 9.0), ("m1:0", 8.0)]
    out = rr.rerank("pricing discount tier", ranked, TEXTS, top_k=2)

    assert [cid for cid, _ in out] == ["m1:0", "ghost:0"]
    assert rr.calls == [[TEXTS["m1:0"]]]


def test_empty_ranking_is_not_an_error():
    rr = FakeCrossEncoder()
    assert rr.rerank("anything", [], TEXTS, top_k=10) == []


def test_top_k_larger_than_the_ranking_is_fine():
    rr = FakeCrossEncoder()
    out = rr.rerank("pricing discount tier", RANKED, TEXTS, top_k=500)
    assert len(out) == len(RANKED)


def test_rerank_timed_reports_milliseconds():
    rr = FakeCrossEncoder()
    out, ms = rr.rerank_timed("pricing", RANKED, TEXTS, top_k=3)

    assert len(out) == len(RANKED)
    assert ms >= 0


# -- specs ------------------------------------------------------------------

def test_none_arm_exists_because_dimension_4_needs_a_baseline():
    assert RR.SPECS["none"] is None
    assert "none" in RR.DEFAULT_ARMS


def test_default_arms_are_all_defined():
    for arm in RR.DEFAULT_ARMS:
        assert arm in RR.SPECS


def test_default_arms_isolate_each_lever():
    # L6@20 vs L6@20/t192 changes only passage length; L6@20/t192 vs L2@20/t192
    # changes only model depth. Without both, a quality drop in the shipped arm
    # cannot be attributed to either lever.
    full = RR.SPECS["L6@20"]
    truncated = RR.SPECS["L6@20/t192"]
    smaller = RR.SPECS["L2@20/t192"]

    assert full.model_id == truncated.model_id and full.top_k == truncated.top_k
    assert full.max_length != truncated.max_length
    assert truncated.max_length == smaller.max_length
    assert truncated.model_id != smaller.model_id


def test_spec_label_names_what_varies():
    assert RR.SPECS["L6@20/t192"].label == "L-6-v2@20/t192"
    assert RR.SPECS["L6@20"].label == "L-6-v2@20"
    assert "onnx-int8" in RR.SPECS["L6@20-onnx"].label


def test_onnx_load_without_an_export_names_the_command():
    with pytest.raises(FileNotFoundError, match="export_onnx_reranker"):
        RR.CrossEncoderReranker.load("nonexistent/model-not-exported", onnx_int8=True)


@pytest.mark.parametrize("logits,expect_second_higher", [
    ([[5.0], [1.0]], False),                     # regressor: one logit
    ([[5.0, 1.0], [1.0, 5.0]], True),            # classifier: [not_rel, rel]
])
def test_both_reranker_head_shapes_are_read_correctly(logits, expect_second_higher):
    # A binary-classifier reranker emits [not_relevant, relevant]. Taking column
    # 0 of that would rank by irrelevance - a silent inversion that reads as a
    # bad reranker rather than as a bug.
    #
    # Tested through the pure function rather than through `score`, so this runs
    # in CI without torch installed.
    scores = RR.relevance_from_logits(np.array(logits, dtype=np.float32))

    assert bool(scores[1] > scores[0]) is expect_second_higher


def test_a_three_logit_head_is_refused_rather_than_guessed():
    # Which column means "relevant" cannot be inferred, and picking one would
    # produce a ranking that looks fine and is arbitrary.
    with pytest.raises(ValueError, match="expected 1"):
        RR.relevance_from_logits(np.zeros((2, 3), dtype=np.float32))


def test_a_flat_logit_array_is_passed_through():
    out = RR.relevance_from_logits(np.array([3.0, 1.0], dtype=np.float32))
    assert out.tolist() == [3.0, 1.0]
