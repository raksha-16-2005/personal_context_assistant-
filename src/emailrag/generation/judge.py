"""Scoring generated answers: groundedness, citation accuracy, relevance.

The structural checks in `synthesize.py` catch what is visible without reading
anything - a `[7]` when six sources were supplied, a sentence asserting something
with no citation. They cannot catch the failure that matters most: a citation that
points at a real source which does not actually support the claim. That needs
somebody to read both, and this module is that reader.

**The judge is a different model from the generator, and that is not optional.**
A model asked whether its own answer is grounded is being asked whether it made a
mistake, and it will say no. Judge and generator are separate clients here, and
`judge_model` is recorded on every verdict so a results file can never be read as
though one model graded itself.

**Every claim is judged against its own cited sources, one at a time.** Handing a
judge the whole answer and the whole context and asking "is this grounded" produces
a single number with no locus - unactionable when it is low, unverifiable when it is
high. Sentence-level verdicts say *which* claim failed and against *which* source.

**Nothing here is trusted without calibration.** An LLM judge is an instrument, and
an uncalibrated instrument reports precision it has not earned. `cohens_kappa`
scores the judge against hand labels; until that number exists, judge scores are
one model's opinion of another's and are labelled as such. κ below about 0.6 means
the judge is measuring itself, not the system.

Refusal is not judged. It is already a machine-checkable fact (the
`INSUFFICIENT_CONTEXT` sentinel), and asking a model to confirm a string comparison
would add cost and noise to something already certain.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm.client import LLM, LLMError
from .synthesize import Answer, Citation, _CITATION

# The judge should be at least as strong as the generator. An under-powered judge
# produces a low kappa that says more about the judge than about the system.
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

VERDICTS = ("supported", "partially_supported", "unsupported", "not_a_claim")

GROUNDED_SYSTEM = (
    "You check whether a claim is supported by an email excerpt. You are strict: "
    "a claim is supported only if the excerpt states it or directly entails it. "
    "Plausibility is not support. You return JSON and nothing else."
)

GROUNDED_PROMPT = """\
Claim: {claim}

Cited excerpt(s):
{sources}

Does the excerpt support the claim?

  supported            the excerpt states or directly entails the claim
  partially_supported  part of the claim is supported, part is not
  unsupported          the excerpt does not support it, even if it sounds plausible
  not_a_claim          the sentence asserts nothing checkable (a connective, a
                       statement about the excerpts themselves, a question)

Return JSON only: {{"verdict": "...", "why": "one short clause"}}"""

RELEVANCE_SYSTEM = (
    "You judge whether an answer addresses the question that was asked. You return "
    "JSON and nothing else."
)

RELEVANCE_PROMPT = """\
Question: {question}

Answer: {answer}

Does the answer address the question? Judge only relevance - whether the answer is
*true* is a separate question you are not being asked.

  0  does not address it at all
  1  addresses part of it, or answers a nearby question
  2  addresses it directly

Return JSON only: {{"score": 0|1|2, "why": "one short clause"}}"""


@dataclass(slots=True)
class ClaimVerdict:
    sentence: str
    cited: list[int]
    verdict: str
    why: str = ""
    judge_model: str = ""

    @property
    def is_supported(self) -> bool:
        return self.verdict == "supported"

    @property
    def counts_toward_groundedness(self) -> bool:
        # A connective or a statement about the sources is not a claim, so it can
        # neither be grounded nor ungrounded. Counting them would dilute the metric
        # with sentences that were never at risk.
        return self.verdict != "not_a_claim"


@dataclass
class AnswerScore:
    question: str
    refused: bool = False
    claims: list[ClaimVerdict] = field(default_factory=list)
    relevance: int | None = None
    relevance_why: str = ""
    invalid_citations: list[int] = field(default_factory=list)
    uncited_claims: int = 0
    judge_model: str = ""
    llm_calls: int = 0
    error: str = ""

    @property
    def n_claims(self) -> int:
        return sum(1 for c in self.claims if c.counts_toward_groundedness)

    @property
    def groundedness(self) -> float | None:
        """Fraction of checkable claims the judge found supported.

        `partially_supported` counts as half. A claim that is half right is not a
        failure and is not a success, and forcing it either way is the kind of
        rounding that makes a metric look cleaner than the thing it measures.
        """
        if not self.n_claims:
            return None
        score = sum(1.0 if c.verdict == "supported"
                    else 0.5 if c.verdict == "partially_supported"
                    else 0.0
                    for c in self.claims if c.counts_toward_groundedness)
        return score / self.n_claims

    @property
    def citation_accuracy(self) -> float | None:
        """Of the claims that cited something, how many cited something that
        supports them.

        Distinct from groundedness: an answer can be perfectly grounded in the
        retrieved context while attributing each claim to the wrong source, and a
        reader who follows the citation would find the wrong message.
        """
        cited = [c for c in self.claims if c.counts_toward_groundedness and c.cited]
        if not cited:
            return None
        return sum(1 for c in cited if c.is_supported) / len(cited)


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip())
            if len(s.strip()) > 3]


class GenerationJudge:
    """Scores answers with a model that did not write them."""

    def __init__(self, llm: LLM | None = None, model: str | None = None) -> None:
        # A separate client, and a separately named model. See the module docstring.
        self._llm = llm
        self._model = model or DEFAULT_JUDGE_MODEL

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(model=self._model)
        return self._llm

    @property
    def judge_model(self) -> str:
        return getattr(self.llm, "model", self._model)

    def score(self, answer: Answer, question: str | None = None,
              judge_relevance: bool = True) -> AnswerScore:
        question = question or answer.question
        score = AnswerScore(question=question, refused=answer.refused,
                            judge_model=self.judge_model,
                            invalid_citations=list(answer.invalid_citations),
                            uncited_claims=len(answer.uncited_sentences))

        if answer.error:
            score.error = answer.error
            return score
        if answer.refused:
            # Refusal is already a machine-checkable fact. Asking a model to
            # confirm a string comparison would add cost and noise to a certainty.
            return score

        by_number = {c.n: c for c in answer.citations}
        for sentence in split_claims(answer.text):
            cited = sorted({int(m.group(1)) for m in _CITATION.finditer(sentence)}
                           & set(by_number))
            verdict = self._judge_claim(sentence, [by_number[n] for n in cited])
            verdict.cited = cited
            score.claims.append(verdict)
            score.llm_calls += 1

        if judge_relevance:
            score.relevance, score.relevance_why = self._judge_relevance(
                question, answer.text)
            score.llm_calls += 1
        return score

    def _judge_claim(self, sentence: str, sources: list[Citation]) -> ClaimVerdict:
        if not sources:
            # No citation to check. This is the structural `uncited` case, already
            # counted; there is nothing for a judge to compare against, and
            # inventing a verdict here would double-count it as ungrounded.
            return ClaimVerdict(sentence=sentence, cited=[], verdict="not_a_claim",
                                why="no citation to check against",
                                judge_model=self.judge_model)

        rendered = "\n\n".join(
            f"[{c.n}] From: {c.sender} Date: {c.date} Subject: {c.subject}\n{c.text}"
            for c in sources)
        try:
            data = self.llm.json_complete(
                GROUNDED_PROMPT.format(claim=sentence, sources=rendered),
                system=GROUNDED_SYSTEM, max_tokens=250)
            verdict = str((data or {}).get("verdict", "")).strip().lower()
            why = str((data or {}).get("why", ""))[:200]
            if verdict not in VERDICTS:
                raise ValueError(f"judge returned {verdict!r}")
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            # A judge failure must not silently become a pass. It is recorded as
            # unsupported with the reason, which is the conservative direction.
            return ClaimVerdict(sentence=sentence, cited=[], verdict="unsupported",
                                why=f"judge failed: {type(exc).__name__}: {exc}",
                                judge_model=self.judge_model)
        return ClaimVerdict(sentence=sentence, cited=[], verdict=verdict, why=why,
                            judge_model=self.judge_model)

    def _judge_relevance(self, question: str, answer: str) -> tuple[int | None, str]:
        try:
            data = self.llm.json_complete(
                RELEVANCE_PROMPT.format(question=question, answer=answer),
                system=RELEVANCE_SYSTEM, max_tokens=200)
            raw = (data or {}).get("score")
            value = int(raw)
            if value not in (0, 1, 2):
                raise ValueError(f"relevance {value} out of range")
            return value, str((data or {}).get("why", ""))[:200]
        except (LLMError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, f"judge failed: {type(exc).__name__}: {exc}"


# -- corpus-level metrics ----------------------------------------------------

@dataclass
class GenerationReport:
    scores: list[AnswerScore] = field(default_factory=list)
    n_unanswerable: int = 0
    n_refused_on_unanswerable: int = 0
    n_refused_on_answerable: int = 0
    judge_model: str = ""

    @property
    def refusal_rate(self) -> float | None:
        """Refusals on the unanswerable controls - the metric those ten queries
        exist for. Reported against the controls only: a refusal on an answerable
        query is a different failure and is counted separately."""
        if not self.n_unanswerable:
            return None
        return self.n_refused_on_unanswerable / self.n_unanswerable

    def _mean(self, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.scores]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    @property
    def groundedness(self) -> float | None:
        return self._mean("groundedness")

    @property
    def citation_accuracy(self) -> float | None:
        return self._mean("citation_accuracy")

    @property
    def relevance(self) -> float | None:
        rel = [s.relevance for s in self.scores if s.relevance is not None]
        return sum(rel) / len(rel) / 2 if rel else None      # normalised to [0,1]

    @property
    def fabricated_citation_rate(self) -> float | None:
        scored = [s for s in self.scores if not s.refused]
        if not scored:
            return None
        return sum(1 for s in scored if s.invalid_citations) / len(scored)

    def render(self, kappa: float | None = None) -> str:
        def fmt(value):
            return f"{value:.3f}" if value is not None else "—"

        lines = [
            "| Metric | Value | n |",
            "|---|---|---|",
            f"| Groundedness | {fmt(self.groundedness)} "
            f"| {sum(s.n_claims for s in self.scores)} claims |",
            f"| Citation accuracy | {fmt(self.citation_accuracy)} "
            f"| {sum(1 for s in self.scores for c in s.claims if c.cited)} cited claims |",
            f"| Answer relevance | {fmt(self.relevance)} "
            f"| {sum(1 for s in self.scores if s.relevance is not None)} answers |",
            f"| Refusal rate on unanswerable controls | {fmt(self.refusal_rate)} "
            f"| {self.n_unanswerable} controls |",
            f"| Fabricated-citation rate | {fmt(self.fabricated_citation_rate)} "
            f"| {sum(1 for s in self.scores if not s.refused)} answers |",
            f"| Refused an answerable query | {self.n_refused_on_answerable} "
            f"| {len(self.scores) - self.n_unanswerable} answerable |",
            "",
        ]
        if kappa is None:
            lines.append(
                "**Uncalibrated.** No hand labels have been compared against this "
                f"judge ({self.judge_model}), so every row above is one model's "
                "opinion of another model's output. Hand-label 50 answers and "
                "report Cohen's κ before quoting these numbers.")
        else:
            grade = ("substantial" if kappa >= 0.6 else
                     "moderate" if kappa >= 0.4 else "poor")
            lines.append(
                f"Judge calibration: Cohen's κ = **{kappa:.3f}** ({grade}) against "
                f"hand labels. Judge model: {self.judge_model}.")
            if kappa < 0.6:
                lines.append(
                    "κ below 0.6 means the judge disagrees with a human often "
                    "enough that these scores describe the judge as much as the "
                    "system. Treat them as indicative only.")
        return "\n".join(lines)


def cohens_kappa(judge: list[str], human: list[str]) -> float:
    """Cohen's κ between two raters over the same items.

    Chance-corrected on purpose: raw agreement is meaningless when one label
    dominates, and "supported" will dominate. A judge that says `supported` to
    everything reaches 85% raw agreement on an 85%-supported set and κ ≈ 0.
    """
    if len(judge) != len(human):
        raise ValueError(f"{len(judge)} judge labels vs {len(human)} human labels")
    n = len(judge)
    if n == 0:
        raise ValueError("no labels to compare")

    labels = sorted(set(judge) | set(human))
    observed = sum(1 for a, b in zip(judge, human) if a == b) / n
    expected = sum((judge.count(l) / n) * (human.count(l) / n) for l in labels)
    if expected >= 1.0:
        # Both raters used a single identical label for everything. Agreement is
        # total and chance agreement is also total, so κ is undefined rather than
        # perfect - reporting 1.0 here would be the most flattering possible lie.
        return float("nan")
    return (observed - expected) / (1 - expected)
