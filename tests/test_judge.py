from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.generation.judge import (
    AnswerScore,
    ClaimVerdict,
    GenerationJudge,
    GenerationReport,
    cohens_kappa,
    split_claims,
)
from emailrag.generation.synthesize import INSUFFICIENT, Answer, Citation


class FakeJudgeLLM:
    """Returns a queued verdict per call, so multi-claim answers are checkable."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.model = "fake-judge"
        self.prompts: list[str] = []

    def json_complete(self, prompt, system="", max_tokens=2048, variant=""):
        self.prompts.append(prompt)
        if not self.payloads:
            return {"verdict": "supported", "why": "default"}
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _sources(n=2):
    return [Citation(n=i + 1, message_id=f"m{i+1}", sender=f"a{i+1}@x.com",
                     date="2001-05-01", subject=f"s{i+1}", text=f"body {i+1}")
            for i in range(n)]


def _answer(text, **kw):
    from emailrag.generation.synthesize import parse_citations, uncited_claim_sentences
    sources = kw.pop("citations", _sources())
    cited, invalid = parse_citations(text, len(sources))
    return Answer(question=kw.pop("question", "what happened"), text=text,
                  citations=sources, cited_numbers=cited, invalid_citations=invalid,
                  uncited_sentences=uncited_claim_sentences(text), **kw)


# -- claim splitting --------------------------------------------------------

def test_claims_split_on_sentence_boundaries():
    assert len(split_claims("First claim [1]. Second claim [2].")) == 2


def test_tiny_fragments_are_not_claims():
    # The judge's own `not_a_claim` verdict handles borderline sentences; this
    # threshold only keeps punctuation noise out of the call budget.
    assert split_claims("No. Ok. A.") == []
    assert split_claims("") == []


# -- groundedness -----------------------------------------------------------

def test_a_supported_claim_scores_grounded():
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "supported", "why": "the excerpt states it"},
        {"score": 2, "why": "direct"}))
    score = judge.score(_answer("The tier is retroactive [1]."))

    assert score.groundedness == 1.0
    assert score.citation_accuracy == 1.0
    assert score.judge_model == "fake-judge"


def test_an_unsupported_claim_scores_zero():
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "unsupported", "why": "plausible but absent"},
        {"score": 2, "why": "x"}))
    score = judge.score(_answer("The tier is retroactive [1]."))

    assert score.groundedness == 0.0
    assert score.citation_accuracy == 0.0


def test_partial_support_counts_as_half():
    # A claim that is half right is neither a failure nor a success, and forcing it
    # either way is the kind of rounding that makes a metric look cleaner than what
    # it measures.
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "partially_supported", "why": "half of it"},
        {"score": 2, "why": "x"}))
    score = judge.score(_answer("The tier is retroactive and legal signed off [1]."))

    assert score.groundedness == 0.5


def test_non_claims_are_excluded_from_the_denominator():
    # A connective was never at risk of being ungrounded; counting it would dilute
    # the metric.
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "supported", "why": "yes"},
        {"verdict": "not_a_claim", "why": "a statement about the excerpts"},
        {"score": 2, "why": "x"}))
    score = judge.score(_answer(
        "The tier is retroactive [1]. The excerpts do not name who approved it [2]."))

    assert score.n_claims == 1
    assert score.groundedness == 1.0


def test_each_claim_is_judged_against_only_its_own_cited_sources():
    # Handing a judge the whole answer and the whole context yields one number with
    # no locus - unactionable when low, unverifiable when high.
    llm = FakeJudgeLLM({"verdict": "supported", "why": "a"},
                       {"verdict": "supported", "why": "b"},
                       {"score": 2, "why": "x"})
    GenerationJudge(llm).score(_answer("Claim one [1]. Claim two [2]."))

    assert "[1]" in llm.prompts[0] and "[2]" not in llm.prompts[0]
    assert "[2]" in llm.prompts[1] and "[1]" not in llm.prompts[1]


def test_an_uncited_sentence_is_not_double_counted_as_ungrounded():
    # It is already counted structurally; there is nothing for a judge to compare
    # it against, and inventing a verdict would count it twice.
    judge = GenerationJudge(FakeJudgeLLM({"score": 2, "why": "x"}))
    score = judge.score(_answer("A confident assertion with no source at all."))

    assert score.n_claims == 0
    assert score.groundedness is None
    assert score.uncited_claims == 1


# -- citation accuracy ------------------------------------------------------

def test_citation_accuracy_only_counts_claims_that_cited_something():
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "supported", "why": "a"},
        {"score": 2, "why": "x"}))
    score = judge.score(_answer(
        "Cited and correct [1]. An assertion with no citation whatsoever here."))

    assert score.citation_accuracy == 1.0        # the uncited one is not counted


def test_citation_accuracy_is_distinct_from_groundedness():
    # An answer can be grounded in the retrieved context while attributing each
    # claim to the wrong source - and a reader following the citation lands wrong.
    judge = GenerationJudge(FakeJudgeLLM(
        {"verdict": "unsupported", "why": "source 1 does not say this"},
        {"score": 2, "why": "x"}))
    score = judge.score(_answer("True of the corpus but wrongly attributed [1]."))

    assert score.citation_accuracy == 0.0


# -- refusal ----------------------------------------------------------------

def test_a_refusal_costs_no_judge_calls():
    # Refusal is already a machine-checkable fact; confirming a string comparison
    # with a model would add cost and noise to a certainty.
    llm = FakeJudgeLLM({"verdict": "supported"})
    score = GenerationJudge(llm).score(_answer(INSUFFICIENT, refused=True))

    assert score.refused
    assert score.llm_calls == 0
    assert llm.prompts == []


def test_a_generation_error_is_carried_not_judged():
    llm = FakeJudgeLLM()
    score = GenerationJudge(llm).score(_answer("", error="LLMError: quota"))

    assert score.error and score.llm_calls == 0
    assert llm.prompts == []


# -- judge failures ---------------------------------------------------------

def test_a_judge_failure_is_conservative_not_a_free_pass():
    from emailrag.llm.client import LLMError
    judge = GenerationJudge(FakeJudgeLLM(LLMError("quota"), {"score": 2, "why": "x"}))
    score = judge.score(_answer("A claim [1]."))

    assert score.groundedness == 0.0
    assert "judge failed" in score.claims[0].why


def test_a_nonsense_verdict_is_treated_as_a_failure():
    judge = GenerationJudge(FakeJudgeLLM({"verdict": "vibes"}, {"score": 2}))
    score = judge.score(_answer("A claim [1]."))

    assert score.claims[0].verdict == "unsupported"


def test_a_relevance_failure_leaves_the_score_none_rather_than_zero():
    from emailrag.llm.client import LLMError
    judge = GenerationJudge(FakeJudgeLLM({"verdict": "supported"}, LLMError("down")))
    score = judge.score(_answer("A claim [1]."))

    assert score.relevance is None
    assert "judge failed" in score.relevance_why


def test_an_out_of_range_relevance_score_is_rejected():
    judge = GenerationJudge(FakeJudgeLLM({"verdict": "supported"}, {"score": 7}))
    assert judge.score(_answer("A claim [1].")).relevance is None


def test_relevance_can_be_skipped():
    llm = FakeJudgeLLM({"verdict": "supported"})
    score = GenerationJudge(llm).score(_answer("A claim [1]."), judge_relevance=False)

    assert score.relevance is None
    assert score.llm_calls == 1


# -- the report -------------------------------------------------------------

def _score(grounded=1.0, cited=True, relevance=2, refused=False, invalid=()):
    verdict = ("supported" if grounded == 1.0
               else "partially_supported" if grounded == 0.5 else "unsupported")
    return AnswerScore(
        question="q", refused=refused, relevance=relevance,
        invalid_citations=list(invalid),
        claims=[ClaimVerdict("s", [1] if cited else [], verdict)],
        judge_model="fake-judge")


def test_the_report_averages_over_answers():
    report = GenerationReport(scores=[_score(1.0), _score(0.0)],
                              judge_model="fake-judge")
    assert report.groundedness == 0.5
    assert report.citation_accuracy == 0.5


def test_relevance_is_normalised_to_zero_one():
    report = GenerationReport(scores=[_score(relevance=2), _score(relevance=0)])
    assert report.relevance == 0.5


def test_refusal_rate_is_measured_against_the_controls_only():
    # A refusal on an answerable query is a different failure and is counted
    # separately; folding it in would make the controls' metric uninterpretable.
    report = GenerationReport(
        scores=[_score(refused=True), _score()],
        n_unanswerable=2, n_refused_on_unanswerable=1, n_refused_on_answerable=1)

    assert report.refusal_rate == 0.5
    assert report.n_refused_on_answerable == 1


def test_refusal_rate_is_none_without_controls():
    assert GenerationReport(scores=[_score()]).refusal_rate is None


def test_fabricated_citation_rate_ignores_refusals():
    report = GenerationReport(scores=[_score(invalid=[9]), _score(),
                                      _score(refused=True)])
    assert report.fabricated_citation_rate == 0.5


def test_an_uncalibrated_report_says_so_loudly():
    # An LLM judge is an instrument, and an uncalibrated instrument reports
    # precision it has not earned.
    rendered = GenerationReport(scores=[_score()], judge_model="fake-judge").render()

    assert "Uncalibrated" in rendered
    assert "one model's opinion of another" in rendered


def test_a_calibrated_report_states_kappa_and_grades_it():
    rendered = GenerationReport(scores=[_score()], judge_model="j").render(kappa=0.71)
    assert "0.710" in rendered and "substantial" in rendered


def test_a_low_kappa_carries_a_warning():
    rendered = GenerationReport(scores=[_score()], judge_model="j").render(kappa=0.35)
    assert "poor" in rendered
    assert "describe the judge as much as the system" in rendered


# -- kappa ------------------------------------------------------------------

def test_perfect_disagreement_free_agreement_is_near_zero():
    # A judge that says "supported" to everything reaches 85% raw agreement on an
    # 85%-supported set. Chance correction is what stops that reading as skill.
    judge = ["supported"] * 10
    human = ["supported"] * 9 + ["unsupported"]

    assert cohens_kappa(judge, human) == pytest.approx(0.0)


def test_perfect_agreement_with_label_variety_is_one():
    judge = ["supported", "unsupported", "supported", "unsupported"]
    assert cohens_kappa(judge, judge) == pytest.approx(1.0)


def test_total_agreement_on_a_single_label_is_undefined_not_perfect():
    # Reporting 1.0 here would be the most flattering possible lie: chance
    # agreement is also total, so there is no skill to measure.
    assert math.isnan(cohens_kappa(["supported"] * 5, ["supported"] * 5))


def test_systematic_disagreement_is_negative():
    judge = ["supported", "supported", "unsupported", "unsupported"]
    human = ["unsupported", "unsupported", "supported", "supported"]
    assert cohens_kappa(judge, human) < 0


def test_mismatched_label_counts_are_rejected():
    with pytest.raises(ValueError, match="judge labels vs"):
        cohens_kappa(["supported"], ["supported", "unsupported"])


def test_empty_labels_are_rejected():
    with pytest.raises(ValueError, match="no labels"):
        cohens_kappa([], [])
