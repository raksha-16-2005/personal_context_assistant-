from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.extraction import dates as D
from emailrag.extraction.extract import CommitmentExtractor
from emailrag.extraction.metrics import compare_arms, same_commitment, score_dates
from emailrag.extraction.schema import KINDS, Commitment, validate

# A Tuesday. Chosen because "next Thursday" is maximally ambiguous mid-week.
TUE = date(2001, 10, 30)
FRI = date(2001, 11, 2)


def r(phrase, sent=TUE):
    return D.resolve(phrase, sent)


# -- weekdays ---------------------------------------------------------------

def test_a_bare_weekday_is_the_next_one_strictly_ahead():
    assert r("due Thursday").value == date(2001, 11, 1)


def test_the_same_weekday_as_the_send_date_means_next_week_not_today():
    # A mail sent Tuesday saying "by Tuesday" cannot mean the moment it was sent.
    assert r("by Tuesday").value == date(2001, 11, 6)


def test_this_weekday_is_the_nearer_one():
    assert r("this Thursday").value == date(2001, 11, 1)


def test_next_weekday_is_the_following_calendar_week():
    # The convention: "this Thursday" is the idiom for the nearer one, so "next"
    # takes the following week.
    out = r("next Thursday")
    assert out.value == date(2001, 11, 8)


def test_next_weekday_is_flagged_ambiguous_and_carries_the_other_reading():
    # US business usage does not converge on this. Scoring it strictly would
    # measure the annotator's convention, not the extractor.
    out = r("next Thursday")
    assert out.ambiguous is True
    assert out.alternative == date(2001, 11, 1)


def test_next_weekday_is_not_ambiguous_when_both_readings_agree():
    # Sent Friday, "next Monday" is the same date under either convention.
    out = r("next Monday", sent=FRI)
    assert out.value == date(2001, 11, 5)
    assert out.ambiguous is False


@pytest.mark.parametrize("phrase,expected", [
    ("mon", date(2001, 11, 5)), ("tues", date(2001, 11, 6)),
    ("wed", date(2001, 10, 31)), ("thurs", date(2001, 11, 1)),
    ("fri", date(2001, 11, 2)),
])
def test_abbreviated_weekdays(phrase, expected):
    assert r(f"by {phrase}").value == expected


# -- relative days ----------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("today", TUE), ("by EOD", TUE), ("end of day", TUE), ("by COB", TUE),
    ("tomorrow", date(2001, 10, 31)), ("day after tomorrow", date(2001, 11, 1)),
    ("yesterday", date(2001, 10, 29)),
])
def test_day_relative_phrases(phrase, expected):
    assert r(phrase).value == expected


@pytest.mark.parametrize("phrase,expected", [
    ("in 3 days", date(2001, 11, 2)),
    ("in two weeks", date(2001, 11, 13)),
    ("in a week", date(2001, 11, 6)),
    ("three days from now", date(2001, 11, 2)),
])
def test_offset_phrases(phrase, expected):
    assert r(phrase).value == expected


def test_business_days_skip_the_weekend():
    # Tuesday + 5 business days is the following Tuesday, not the Sunday.
    assert r("in 5 business days").value == date(2001, 11, 6)
    assert r("within 3 working days").value == date(2001, 11, 2)


def test_in_one_month_clamps_to_a_short_month():
    # 31 January + one month is 28 February, not 3 March.
    assert r("in 1 month", sent=date(2001, 1, 31)).value == date(2001, 2, 28)


def test_in_one_month_handles_a_leap_year():
    assert r("in 1 month", sent=date(2000, 1, 31)).value == date(2000, 2, 29)


# -- period ends ------------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("end of week", FRI), ("by EOW", FRI),
    ("end of next week", date(2001, 11, 9)),
    ("end of the month", date(2001, 10, 31)),
    ("end of next month", date(2001, 11, 30)),
    ("end of quarter", date(2001, 12, 31)),
])
def test_period_ends(phrase, expected):
    assert r(phrase).value == expected


def test_end_of_week_on_a_friday_is_that_friday():
    assert r("end of week", sent=FRI).value == FRI


def test_vaguer_periods_carry_coarser_precision():
    # "sometime next month" is not a day, and treating it as one manufactures
    # precision the message never had.
    assert r("next month").precision == D.PRECISION_MONTH
    assert r("next week").precision == D.PRECISION_WEEK
    assert r("next quarter").precision == D.PRECISION_QUARTER
    assert r("Thursday").precision == D.PRECISION_DAY


def test_end_of_february_is_correct_in_a_leap_year():
    assert r("end of month", sent=date(2000, 2, 3)).value == date(2000, 2, 29)


# -- explicit dates ---------------------------------------------------------

@pytest.mark.parametrize("phrase,expected", [
    ("by November 8", date(2001, 11, 8)),
    ("by Nov. 8", date(2001, 11, 8)),
    ("by 8 November", date(2001, 11, 8)),
    ("by the 8th of November", date(2001, 11, 8)),
    ("on 2001-11-08", date(2001, 11, 8)),
    ("by 11/8", date(2001, 11, 8)),
    ("by 11/8/01", date(2001, 11, 8)),
    ("by November 8, 2002", date(2002, 11, 8)),
])
def test_explicit_dates(phrase, expected):
    assert r(phrase).value == expected


def test_a_month_day_well_in_the_past_rolls_to_next_year():
    # A mail sent 30 December saying "by January 3" means the coming January.
    out = D.resolve("by January 3", date(2000, 12, 30))
    assert out.value == date(2001, 1, 3)
    assert out.rolled_year is True


def test_a_month_day_slightly_in_the_past_does_not_roll():
    # Within tolerance: an overdue deadline is a real thing an email can mention.
    out = D.resolve("the October 15 filing", date(2001, 10, 30))
    assert out.value == date(2001, 10, 15)
    assert out.rolled_year is False


def test_a_day_over_12_in_the_first_numeric_slot_is_refused():
    # 13/11 cannot be month/day, and reinterpreting it as day/month would produce
    # a plausible wrong date in a corpus that is entirely US-formatted.
    with pytest.raises(D.UnresolvableDate, match="cannot be assumed"):
        r("by 13/11")


def test_an_impossible_date_is_refused_not_clamped():
    with pytest.raises(D.UnresolvableDate, match="not a real date"):
        r("by February 30")


def test_two_digit_years_land_in_the_right_century():
    assert r("by 1/5/99").value == date(1999, 1, 5)
    assert r("by 1/5/02").value == date(2002, 1, 5)


# -- no date ----------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "", "   ", "before the board meeting", "as soon as possible", "when you can",
])
def test_phrases_with_no_computable_date_are_refused(phrase):
    with pytest.raises(D.UnresolvableDate):
        r(phrase)


def test_try_resolve_returns_none_instead_of_raising():
    # Most messages have no deadline; that is not an exceptional condition.
    assert D.try_resolve("before the board meeting", TUE) is None
    assert D.try_resolve("Thursday", TUE) is not None


# -- resolution is relative to the message, never to now --------------------

def test_resolution_uses_the_send_date_not_today():
    # A 1999-2002 corpus has no "now". Same phrase, two send dates, two answers.
    assert r("Thursday", sent=date(1999, 3, 1)).value == date(1999, 3, 4)
    assert r("Thursday", sent=date(2002, 6, 10)).value == date(2002, 6, 13)


def test_a_datetime_send_date_is_accepted():
    sent = datetime(2001, 10, 30, 14, 30, tzinfo=timezone.utc)
    assert r("tomorrow", sent=sent).value == date(2001, 10, 31)


def test_resolved_utc_is_midnight_of_the_date():
    out = r("tomorrow")
    assert out.resolved_utc == datetime(2001, 10, 31, tzinfo=timezone.utc)
    assert out.iso == "2001-10-31"


# -- the pre-filter ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Can you get me your comments before Thursday?",
    "The redline is due by EOD.",
    "I'll send the summary tomorrow.",
    "Please review the attached MSA.",
    "Deadline is 11/15.",
    "Action item: confirm the Q3 date.",
    "no later than end of business",
])
def test_the_prefilter_passes_commitment_shaped_text(text):
    assert D.looks_like_commitment(text)


@pytest.mark.parametrize("text", [
    "Thanks, that works for me.",
    "FYI - article below about the industry.",
    "Cafeteria menu: Tuesday is taco day.",
    "Happy birthday to Diane!",
    "",
])
def test_the_prefilter_drops_text_with_no_obligation(text):
    assert not D.looks_like_commitment(text)


def test_the_prefilter_is_tuned_for_recall():
    # A false positive costs 25 seconds of CPU; a false negative loses a
    # commitment permanently and invisibly. Borderline text must pass.
    assert D.looks_like_commitment("we need to decide this week")
    assert D.looks_like_commitment("could you sign the form")


# -- schema -----------------------------------------------------------------

def test_a_valid_commitment_has_no_problems():
    c = Commitment(message_id="m1", text="send the redline", kind="deliverable",
                   direction="i_owe", confidence=0.9)
    assert validate(c) == []


def test_a_resolved_date_without_its_phrase_is_rejected():
    # due_at is derived from due_phrase plus the send date. Without the phrase the
    # row cannot be re-resolved or audited, which is what the schema exists for.
    c = Commitment(message_id="m1", text="x", due_at=date(2001, 11, 1))
    assert any("not auditable" in p for p in validate(c))


@pytest.mark.parametrize("kwargs,expected", [
    ({"message_id": ""}, "no message_id"),
    ({"text": "  "}, "empty text"),
    ({"kind": "invented"}, "unknown kind"),
    ({"direction": "sideways"}, "unknown direction"),
    ({"confidence": 1.5}, "outside"),
])
def test_schema_problems_are_named(kwargs, expected):
    base = dict(message_id="m1", text="x", kind="other", direction="unclear")
    problems = validate(Commitment(**{**base, **kwargs}))
    assert any(expected in p for p in problems)


def test_as_row_serialises_dates_for_the_database():
    c = Commitment(message_id="m1", text="x", due_phrase="Thursday",
                   due_at=date(2001, 11, 1), due_alternative=date(2001, 11, 8))
    row = c.as_row()
    assert row["due_at"] == "2001-11-01"
    assert row["due_alternative"] == "2001-11-08"


# -- extraction -------------------------------------------------------------

class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.model = "fake-3b"
        self.last_cached = False
        self.prompts = []

    def json_complete(self, prompt, system="", max_tokens=2048, variant=""):
        self.prompts.append(prompt)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


MESSAGE = {
    "dedup_key": "m1", "sender": "a@x.com", "recipients": "b@x.com",
    "subject": "MSA redline", "body_new": "Can you send comments before Thursday?",
    "date_utc": datetime(2001, 10, 30, tzinfo=timezone.utc),
}


def test_extraction_resolves_the_phrase_in_python_not_the_model():
    # The model returns "before Thursday"; the date is computed here. This is the
    # whole division of labour - models are unreliable at date arithmetic.
    llm = FakeLLM({"commitments": [{
        "text": "send comments on the MSA", "kind": "review", "direction": "they_owe",
        "due_phrase": "before Thursday", "confidence": 0.9, "quote": "Can you send…"}]})
    out = CommitmentExtractor(llm).extract(MESSAGE)

    assert len(out) == 1
    assert out[0].due_phrase == "before Thursday"
    assert out[0].due_at == date(2001, 11, 1)
    assert out[0].model == "fake-3b"


def test_the_prompt_tells_the_model_not_to_compute_dates():
    llm = FakeLLM({"commitments": []})
    CommitmentExtractor(llm).extract(MESSAGE)
    # The send date's weekday is stated so the model can judge whether a phrase is
    # a deadline at all, without being asked to do the arithmetic.
    assert "Tuesday" in llm.prompts[0]


def test_an_unresolvable_phrase_keeps_the_commitment_and_is_counted():
    llm = FakeLLM({"commitments": [{"text": "x", "due_phrase": "before the board meeting"}]})
    ex = CommitmentExtractor(llm)
    out = ex.extract(MESSAGE)

    assert len(out) == 1 and out[0].due_at is None
    assert out[0].due_phrase == "before the board meeting"
    assert ex.stats.unresolvable_phrases == 1


def test_an_unknown_kind_is_coerced_rather_than_dropping_the_commitment():
    # Losing a real commitment to a taxonomy mismatch is a worse error than
    # filing it as "other".
    llm = FakeLLM({"commitments": [{"text": "x", "kind": "escalation"}]})
    out = CommitmentExtractor(llm).extract(MESSAGE)

    assert len(out) == 1 and out[0].kind == "other"
    assert out[0].kind in KINDS


def test_the_prefilter_skips_a_message_without_calling_the_model():
    llm = FakeLLM({"commitments": [{"text": "should never be reached"}]})
    ex = CommitmentExtractor(llm)
    out = ex.extract({**MESSAGE, "subject": "hello", "body_new": "Thanks, appreciated."})

    assert out == []
    assert llm.prompts == []
    assert ex.stats.messages_prefiltered_out == 1
    assert ex.stats.messages_called == 0


def test_the_prefilter_can_be_disabled():
    llm = FakeLLM({"commitments": []})
    ex = CommitmentExtractor(llm, prefilter=False)
    ex.extract({**MESSAGE, "body_new": "Thanks, appreciated."})

    assert ex.stats.messages_called == 1


def test_a_model_failure_is_recorded_and_skipped_not_raised():
    # One pass over the corpus is an overnight job; it must not die on one message.
    from emailrag.llm.client import LLMError
    ex = CommitmentExtractor(FakeLLM(LLMError("ollama down")))
    out = ex.extract(MESSAGE)

    assert out == []
    assert ex.stats.failures == 1
    assert "ollama down" in ex.errors[0]["error"]


def test_a_non_list_response_is_a_failure_not_a_crash():
    ex = CommitmentExtractor(FakeLLM({"commitments": "Thursday"}))
    assert ex.extract(MESSAGE) == []
    assert ex.stats.failures == 1


def test_a_bare_array_response_is_accepted():
    ex = CommitmentExtractor(FakeLLM([{"text": "send it", "due_phrase": "tomorrow"}]))
    out = ex.extract(MESSAGE)
    assert len(out) == 1 and out[0].due_at == date(2001, 10, 31)


def test_empty_commitments_is_a_normal_outcome():
    ex = CommitmentExtractor(FakeLLM({"commitments": []}))
    assert ex.extract(MESSAGE) == []
    assert ex.stats.failures == 0


def test_stats_report_the_prefilter_pass_rate():
    ex = CommitmentExtractor(FakeLLM({"commitments": []}))
    ex.extract(MESSAGE)
    ex.extract({**MESSAGE, "body_new": "Thanks!"})

    assert ex.stats.messages_seen == 2
    assert ex.stats.prefilter_pass_rate == 0.5
    assert "prefiltered out" in ex.stats.render()


# -- metrics ----------------------------------------------------------------

def _c(text="send the redline", due=None, ambiguous=False, alt=None, mid="m1"):
    return Commitment(message_id=mid, text=text, due_phrase="x", due_at=due,
                      due_ambiguous=ambiguous, due_alternative=alt)


def test_exact_dates_score_exact():
    score = score_dates([(_c(due=date(2001, 11, 1)), _c(due=date(2001, 11, 1)))])
    assert score.n == 1 and score.exact == 1 and score.within_1d == 1


def test_off_by_one_day_counts_within_1d_but_not_exact():
    score = score_dates([(_c(due=date(2001, 11, 2)), _c(due=date(2001, 11, 1)))])
    assert score.exact == 0 and score.within_1d == 1


def test_an_ambiguous_phrase_right_under_the_other_convention_scores_either_reading():
    # "next Thursday" has no single correct answer. Strict scoring here would
    # measure the annotator's convention, not the extractor.
    got = _c(due=date(2001, 11, 8), ambiguous=True, alt=date(2001, 11, 1))
    score = score_dates([(got, _c(due=date(2001, 11, 1)))])

    assert score.exact == 0
    assert score.either_reading == 1
    assert score.ambiguous == 1


def test_a_missing_date_counts_against_the_denominator():
    score = score_dates([(_c(due=None), _c(due=date(2001, 11, 1)))])
    assert score.n == 1 and score.missing == 1 and score.exact == 0


def test_a_spurious_date_is_counted_outside_the_denominator():
    # Gold says there is no deadline; inventing one is a different error from
    # dating a real one wrongly.
    score = score_dates([(_c(due=date(2001, 11, 1)), _c(due=None))])
    assert score.spurious == 1 and score.n == 0


def test_no_date_on_either_side_is_not_scored():
    assert score_dates([(_c(), _c())]).n == 0


def test_matching_is_fuzzy_because_two_models_word_things_differently():
    assert same_commitment(_c("send the redlined MSA to legal"),
                           _c("send redlined MSA"))
    assert not same_commitment(_c("send the redline"), _c("book the conference room"))


def test_matching_never_crosses_messages():
    assert not same_commitment(_c("send the redline", mid="m1"),
                               _c("send the redline", mid="m2"))


def test_arm_comparison_reports_agreement_not_accuracy():
    local = [_c("send the redline"), _c("book the room", mid="m2")]
    ceiling = [_c("send redline to legal"), _c("call the counterparty", mid="m3")]

    cmp = compare_arms(local, ceiling, local_model="qwen", ceiling_model="haiku")

    assert cmp.both == 1
    assert cmp.local_only == 1 and cmp.ceiling_only == 1
    assert cmp.agreement == pytest.approx(1 / 3)
    assert "agreement, not accuracy" in cmp.render()


def test_commitments_from_unlabelled_messages_are_not_scored_as_spurious():
    # Otherwise every extraction over unlabelled data counts as an error.
    local = [_c("send the redline", due=date(2001, 11, 1), mid="unlabelled")]
    gold = {"m1": [_c("send the redline", due=date(2001, 11, 1))]}

    cmp = compare_arms(local, [], gold=gold)
    assert cmp.local_dates.n == 0 and cmp.local_dates.spurious == 0
