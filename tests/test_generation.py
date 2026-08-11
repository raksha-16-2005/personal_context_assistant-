from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.generation.synthesize import (
    INSUFFICIENT,
    Answer,
    Citation,
    Synthesizer,
    format_sources,
    parse_citations,
    uncited_claim_sentences,
)
from emailrag.llm.client import LLMError


class FakeLLM:
    def __init__(self, response: str = "", raises: Exception | None = None):
        self.response = response
        self.raises = raises
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.last_model = "fake-model"
        self.last_cached = False

    def complete(self, prompt, system="", max_tokens=2048, variant="") -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self.raises:
            raise self.raises
        return self.response


def _sources(n: int = 3) -> list[Citation]:
    return [Citation(n=i + 1, message_id=f"m{i+1}", sender=f"a{i+1}@x.com",
                     recipients="b@x.com", date=f"2001-05-0{i+1}",
                     subject=f"subject {i+1}", text=f"body of message {i+1}")
            for i in range(n)]


# -- citation parsing -------------------------------------------------------

def test_valid_citations_are_collected_in_order_of_first_use():
    # First-use order is how a reader scans them, so it is the order a UI lists.
    cited, invalid = parse_citations("Claim A [3]. Claim B [1][3]. Claim C [2].", 3)

    assert cited == [3, 1, 2]
    assert invalid == []


def test_out_of_range_citations_are_reported_not_silently_dropped():
    # A [7] when six sources were supplied is a fabricated citation, and it is
    # exactly the failure a reader cannot detect by eye.
    cited, invalid = parse_citations("Real [1]. Invented [7]. Also invented [9].", 3)

    assert cited == [1]
    assert invalid == [7, 9]


def test_zero_is_not_a_valid_citation():
    cited, invalid = parse_citations("Nonsense [0].", 3)
    assert cited == [] and invalid == [0]


def test_no_citations_at_all():
    assert parse_citations("A confident, unsourced assertion.", 3) == ([], [])


# -- uncited claims ---------------------------------------------------------

def test_an_uncited_claim_is_flagged():
    out = uncited_claim_sentences("The tier applies retroactively to all accounts.")
    assert len(out) == 1


def test_a_cited_claim_is_not_flagged():
    assert uncited_claim_sentences("The tier applies retroactively [2].") == []


def test_statements_about_the_sources_are_not_uncited_claims():
    # A partial refusal is the behaviour this system wants most. Flagging it would
    # train a reader to ignore the flag. This exact output came from the real
    # pipeline, asked for the capital of France.
    text = ("Paris is a city in France [5]. The excerpts do not explicitly state "
            "that Paris is the capital of France.")
    assert uncited_claim_sentences(text) == []


@pytest.mark.parametrize("sentence", [
    "The sources do not say who approved the change.",
    "None of the excerpts mention a termination date.",
    "The date is not specified in the provided emails.",
])
def test_partial_refusals_are_recognised_as_meta(sentence):
    assert uncited_claim_sentences(sentence) == []


def test_connectives_and_questions_are_skipped():
    assert uncited_claim_sentences("Two things stand out here:") == []
    assert uncited_claim_sentences("Was the agreement ever countersigned by them?") == []


# -- prompt construction ----------------------------------------------------

def test_sources_include_headers_because_half_the_questions_are_about_who_and_when():
    # A body-only excerpt makes "who owns the Calpine renewal" unanswerable even
    # when the answering message is right there.
    rendered = format_sources(_sources(2))

    assert "From: a1@x.com" in rendered
    assert "Date: 2001-05-01" in rendered
    assert "Subject: subject 1" in rendered
    assert "[2]" in rendered


def test_long_sources_are_clipped():
    long_source = [Citation(n=1, message_id="m1", text="word " * 2000)]
    rendered = format_sources(long_source)

    assert len(rendered) < 2000
    assert rendered.rstrip().endswith("…")


def test_the_as_of_anchor_reaches_the_prompt():
    llm = FakeLLM("Answer [1].")
    Synthesizer(llm).answer("what is due", _sources(), as_of="2001-10-30")

    assert "2001-10-30" in llm.prompts[0]


def test_mailbox_owner_reaches_the_prompt_only_when_given():
    llm = FakeLLM("Answer [1].")
    Synthesizer(llm).answer("how many did I get", _sources(),
                            mailbox_owner="alice@example.com")

    assert "alice@example.com" in llm.prompts[0]
    assert '"I", "me", and "my"' in llm.prompts[0]

    llm2 = FakeLLM("Answer [1].")
    Synthesizer(llm2).answer("how many did I get", _sources())
    assert "alice@example.com" not in llm2.prompts[0]


def test_route_note_reaches_the_prompt_and_forbids_citing_it():
    llm = FakeLLM("Answer [1].")
    Synthesizer(llm).answer(
        "how many did I get today", _sources(),
        route_note="3 non-bulk message(s) received today; 17 promotional message(s) filtered")

    prompt = llm.prompts[0]
    assert "17 promotional message(s) filtered" in prompt
    # The model latched onto "authoritative" phrasing once and echoed it back
    # as a fake "[Authoritative]" citation - see synthesize.py's own comment
    # on `route_note_line`. Asserting the instruction not to cite this line
    # is what would catch a regression back to that wording.
    assert "no bracketed citation" in prompt


def test_no_route_note_line_when_route_note_is_empty():
    llm = FakeLLM("Answer [1].")
    Synthesizer(llm).answer("q", _sources())

    assert "mailbox's date index" not in llm.prompts[0]


def test_the_system_prompt_demands_the_refusal_sentinel():
    llm = FakeLLM("Answer [1].")
    Synthesizer(llm).answer("q", _sources())

    assert INSUFFICIENT in llm.systems[0]


# -- answering --------------------------------------------------------------

def test_a_cited_answer_is_parsed_and_marked_grounded_in_shape():
    llm = FakeLLM("The tier is retroactive [1]. Legal signed off [2].")
    answer = Synthesizer(llm).answer("what about the tier", _sources())

    assert not answer.refused
    assert answer.cited_numbers == [1, 2]
    assert answer.invalid_citations == []
    assert answer.uncited_sentences == []
    assert answer.is_grounded_shape
    assert answer.model == "fake-model"
    assert answer.latency_ms >= 0


def test_the_refusal_sentinel_sets_refused():
    llm = FakeLLM(INSUFFICIENT)
    answer = Synthesizer(llm).answer("who won the 2030 world cup", _sources())

    assert answer.refused
    assert answer.is_grounded_shape          # refusing is a grounded outcome


def test_a_bolded_or_padded_sentinel_still_counts_as_refusal():
    # Models wrap it in markdown regardless of instructions.
    for raw in (f"**{INSUFFICIENT}**", f"  {INSUFFICIENT}  ",
                f"{INSUFFICIENT}\n", INSUFFICIENT.lower()):
        assert Synthesizer(FakeLLM(raw)).answer("q", _sources()).refused


def test_a_prose_apology_is_not_treated_as_a_refusal():
    # Refusal is a sentinel precisely so this is not a string-matching problem.
    # This answer cites nothing and refuses nothing - it should read as ungrounded.
    llm = FakeLLM("I'm sorry, I could not find anything relevant in the emails.")
    answer = Synthesizer(llm).answer("q", _sources())

    assert not answer.refused
    assert not answer.is_grounded_shape


def test_fabricated_citations_are_surfaced_and_break_grounded_shape():
    llm = FakeLLM("Definitely true [9].")
    answer = Synthesizer(llm).answer("q", _sources(3))

    assert answer.invalid_citations == [9]
    assert not answer.is_grounded_shape


def test_no_sources_refuses_without_spending_an_llm_call():
    # Asking a model to answer from an empty context invites it to answer from
    # its own weights instead.
    llm = FakeLLM("this must never be returned")
    answer = Synthesizer(llm).answer("q", [])

    assert answer.refused and answer.text == INSUFFICIENT
    assert llm.prompts == []


def test_only_n_sources_are_sent_to_the_model():
    llm = FakeLLM("Answer [1].")
    answer = Synthesizer(llm, n_sources=2).answer("q", _sources(6))

    assert len(answer.citations) == 2
    assert "[3]" not in llm.prompts[0]


def test_an_llm_failure_is_reported_not_raised():
    # Retrieval still worked; the UI should show the sources and say generation
    # failed, not lose the whole response.
    llm = FakeLLM(raises=LLMError("quota exhausted"))
    answer = Synthesizer(llm).answer("q", _sources())

    assert answer.error and "quota" in answer.error
    assert answer.text == ""
    assert len(answer.citations) == 3        # sources survive


def test_cited_sources_returns_only_what_the_answer_used():
    llm = FakeLLM("Only this one matters [2].")
    answer = Synthesizer(llm).answer("q", _sources(4))

    assert [c.n for c in answer.cited_sources] == [2]


def test_citation_label_is_readable():
    c = Citation(n=3, message_id="m", sender="leslie.hansen@enron.com",
                 date="2000-12-11", subject="Confidentiality Agreement")

    assert c.label == "[3] leslie.hansen, 2000-12-11: Confidentiality Agreement"


def test_citation_label_survives_missing_metadata():
    assert "unknown" in Citation(n=1, message_id="m").label
    assert "(no subject)" in Citation(n=1, message_id="m").label


def test_answer_defaults_are_not_grounded():
    assert not Answer(question="q", text="", citations=[]).is_grounded_shape
