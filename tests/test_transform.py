from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.llm.client import LLMError
from emailrag.query.transform import QueryTransformer, TransformedQuery, identity

QUERY = "what did we decide about the discount tier"


class FakeLLM:
    """Returns canned text, and records what it was asked."""

    def __init__(self, response: str = "", raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.prompts: list[str] = []
        self.last_cached = False

    def complete(self, prompt, system="", max_tokens=2048, variant="") -> str:
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return self.response

    def json_complete(self, prompt, system="", max_tokens=2048, variant=""):
        raw = self.complete(prompt, system, max_tokens, variant)
        return json.loads(raw)


def _t(kind, llm):
    return QueryTransformer(kind, llm=llm)


# -- baseline ---------------------------------------------------------------

def test_none_arm_needs_no_llm_at_all():
    # `--transform none` has to run with no API key: it is the baseline every
    # other arm is measured against.
    xf = QueryTransformer("none")
    out = xf(QUERY)

    assert out.dense_texts == [QUERY] and out.sparse_texts == [QUERY]
    assert out.llm_calls == 0 and out.n_runs == 1
    assert xf.stats.fired == 0


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="unknown transform"):
        QueryTransformer("telepathy")


def test_identity_is_a_single_run():
    assert identity("q").n_runs == 1


# -- hyde -------------------------------------------------------------------

HYDE_DOC = ("From: sara@company.com\nTo: mark@company.com\nDate: 2001-10-30\n"
            "Subject: re: discount tier\n\nWe agreed it applies retroactively.")


def test_hyde_embeds_the_document_but_searches_bm25_with_the_original():
    # An invented sender is a gift to a bi-encoder and pure noise to BM25, which
    # would match the fabricated name literally. Applying one rewrite to both
    # retrievers would confound "does HyDE help" with "does HyDE help BM25".
    llm = FakeLLM(HYDE_DOC)
    out = _t("hyde", llm)(QUERY)

    assert out.dense_texts == [HYDE_DOC]
    assert out.sparse_texts == [QUERY]
    assert out.llm_calls == 1
    assert out.n_runs == 1


def test_hyde_passes_as_of_so_the_document_can_be_dated():
    llm = FakeLLM(HYDE_DOC)
    _t("hyde", llm)(QUERY, as_of="2001-10-30")

    assert "2001-10-30" in llm.prompts[0]


def test_hyde_prompt_asks_for_an_email_not_an_essay():
    # Indexed chunks begin with a literal From/To/Date/Subject header block, and
    # dense retrieval scores style as readily as content.
    llm = FakeLLM(HYDE_DOC)
    _t("hyde", llm)(QUERY)

    prompt = llm.prompts[0]
    assert "From:" in prompt and "Subject:" in prompt


def test_hyde_empty_document_degrades_to_the_original():
    xf = _t("hyde", FakeLLM("   "))
    out = xf(QUERY)

    assert out.degraded and out.dense_texts == [QUERY]
    assert xf.stats.degraded == 1
    assert xf.stats.fired == 0


# -- multi-query ------------------------------------------------------------

def test_multi_query_keeps_the_original_alongside_the_rewrites():
    # A paraphrase can be worse than what the user typed; fusion must never be
    # able to lose a result the baseline would have found.
    llm = FakeLLM(json.dumps({"rewrites": ["discount tier decision",
                                           "tier pricing agreement",
                                           "volume discount outcome"]}))
    out = _t("multi_query", llm)(QUERY)

    assert out.dense_texts[0] == QUERY
    assert len(out.dense_texts) == 4
    assert out.dense_texts == out.sparse_texts
    assert out.n_runs == 4
    assert out.llm_calls == 1        # k rewrites in ONE call, not k calls


def test_multi_query_respects_n_rewrites():
    llm = FakeLLM(json.dumps({"rewrites": ["a", "b", "c", "d", "e"]}))
    xf = QueryTransformer("multi_query", llm=llm, n_rewrites=2)
    out = xf(QUERY)

    assert len(out.dense_texts) == 3          # original + 2
    assert "2 different ways" in llm.prompts[0]


def test_multi_query_tolerates_a_bare_array():
    llm = FakeLLM(json.dumps(["first rewrite", "second rewrite"]))
    out = _t("multi_query", llm)(QUERY)

    assert out.dense_texts == [QUERY, "first rewrite", "second rewrite"]


def test_multi_query_with_no_usable_rewrites_degrades():
    xf = _t("multi_query", FakeLLM(json.dumps({"rewrites": []})))
    out = xf(QUERY)

    assert out.degraded and out.dense_texts == [QUERY]
    assert xf.stats.degraded == 1


def test_malformed_json_degrades_instead_of_killing_the_run():
    # A three-hour benchmark must not die on one malformed rewrite.
    xf = _t("multi_query", FakeLLM("not json at all"))
    out = xf(QUERY)

    assert out.degraded
    assert "JSONDecodeError" in out.raw or "Error" in out.raw


def test_an_llm_error_degrades_instead_of_propagating():
    xf = _t("hyde", FakeLLM(raises=LLMError("quota")))
    out = xf(QUERY)

    assert out.degraded and out.dense_texts == [QUERY]


# -- decomposition ----------------------------------------------------------

def test_decompose_splits_a_multi_hop_query():
    llm = FakeLLM(json.dumps({"multi_hop": True, "sub_questions": [
        "what did legal say about the MSA",
        "when did the steering committee meet"]}))
    out = _t("decompose", llm)(QUERY)

    assert out.dense_texts[0] == QUERY
    assert len(out.dense_texts) == 3
    assert out.n_runs == 3
    assert not out.degraded


def test_a_single_lookup_is_not_a_degradation():
    # "This query is one lookup" is the correct answer for most of the eval set.
    # Counting it as degraded would make the dimension-5 table report a failure
    # rate where there is none - how often decomposition *fires* is the result.
    llm = FakeLLM(json.dumps({"multi_hop": False, "sub_questions": []}))
    xf = _t("decompose", llm)
    out = xf(QUERY)

    assert not out.degraded
    assert out.dense_texts == [QUERY]
    assert xf.stats.degraded == 0
    assert xf.stats.fired == 0
    assert out.llm_calls == 1        # it still cost a call, and that is reported


def test_multi_hop_true_with_one_sub_question_is_treated_as_atomic():
    # One sub-question is the original query with extra words; splitting on it
    # would add a retrieval pass for nothing.
    llm = FakeLLM(json.dumps({"multi_hop": True, "sub_questions": ["only one"]}))
    out = _t("decompose", llm)(QUERY)

    assert out.dense_texts == [QUERY]


def test_decompose_caps_the_number_of_sub_questions():
    llm = FakeLLM(json.dumps({"multi_hop": True,
                              "sub_questions": ["a", "b", "c", "d", "e"]}))
    out = _t("decompose", llm)(QUERY)

    assert len(out.dense_texts) == 4      # original + MAX_SUBQUESTIONS


def test_decompose_rejects_a_bare_array_response():
    # The contract is an object with multi_hop; an array cannot say whether the
    # query was multi-hop at all, so it is a malformed response, not a split.
    xf = _t("decompose", FakeLLM(json.dumps(["a", "b"])))
    out = xf(QUERY)

    assert out.degraded


# -- accounting -------------------------------------------------------------

def test_stats_count_queries_fired_and_calls():
    llm = FakeLLM(json.dumps({"rewrites": ["one", "two"]}))
    xf = _t("multi_query", llm)
    for q in ("a", "b", "c"):
        xf(q)

    assert xf.stats.queries == 3
    assert xf.stats.fired == 3
    assert xf.stats.calls == 3
    assert "3/3 fired" in xf.stats.render()


def test_the_log_records_what_the_model_returned():
    llm = FakeLLM(HYDE_DOC)
    xf = _t("hyde", llm)
    xf(QUERY)

    assert xf.log[0]["kind"] == "hyde"
    assert HYDE_DOC[:20] in xf.log[0]["raw"]


def test_transformed_query_n_runs_is_the_latency_multiplier():
    tq = TransformedQuery("q", ["a", "b", "c"], ["q"], kind="multi_query")
    assert tq.n_runs == 3
