from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.extraction.schema import Commitment
from emailrag.router.classify import (
    CLASS_TO_ROUTE,
    QueryRouter,
    Route,
    RouterDecision,
    classify_rules,
    routes_table,
    score_routes,
)
from emailrag.router.sql import (
    filter_commitments,
    filter_messages_by_date,
    parse_window,
    window_sql,
)

TUE = date(2001, 10, 30)          # a Tuesday


# -- rule-based classification ----------------------------------------------

@pytest.mark.parametrize("query", [
    "what's due this week",
    "what is outstanding next month",
    "anything overdue",
    "what are the deadlines this quarter",
    "what is due by November 8",
])
def test_pure_date_questions_route_to_sql(query):
    # These have no semantically similar passage: the answer is a date comparison,
    # and embedding the question retrieves messages that talk about deadlines
    # rather than messages with deadlines in the window.
    assert classify_rules(query).name == "sql"


@pytest.mark.parametrize("query", [
    "what came in today",
    "anything arriving tomorrow",
])
def test_bare_today_and_tomorrow_route_to_sql(query):
    # "today"/"tomorrow" alone used to fall through to pure retrieval, which
    # has no notion of "today" - parse_window already resolves both into a
    # real window (see test_router's _WINDOWS coverage), the rules just never
    # told the router to use it. These two have no content noun ("mail",
    # "emails") for _SEMANTIC to catch, so they stay pure date questions.
    assert classify_rules(query).name == "sql"


@pytest.mark.parametrize("query", [
    "what emails did I get today",
    "summarize today's mail",
    "what all mail did i get last week give me an abstract",
])
def test_today_or_this_week_plus_a_mail_summary_request_routes_to_both(query):
    # The bug this guards against: these used to fall through to pure "sql"
    # too, since nothing told the rules that "mail"/"summary"/"abstract"
    # signal real content, not just a date filter. A pure "sql" route never
    # runs retrieval, so a user asking this got back only whatever the date
    # arms found (often an unrelated due-today commitment, or nothing) with
    # no actual mail content - and, being neither "mail I received" nor
    # substantive, the generator correctly refused rather than answer from
    # it, which read as the system ignoring an otherwise-reasonable question.
    assert classify_rules(query).name == "both"


@pytest.mark.parametrize("query", [
    "how many commitments are open",
    "how much was the discount",
    "list all deadlines",
    "count of messages per week",
])
def test_aggregates_route_to_sql_whatever_else_they_contain(query):
    # A ranked list cannot answer "how many". The top of a ranked list is not a
    # count.
    assert classify_rules(query).name == "sql"


def test_an_aggregate_beats_a_semantic_signal():
    out = classify_rules("how many messages discuss the pricing model")
    assert out.name == "sql"


@pytest.mark.parametrize("query", [
    "what's most urgent",
    "what's the most important thing right now",
])
def test_bare_superlatives_are_not_aggregate_phrasing(query):
    # "most"/"fewest"/"busiest"/average used to be bare _AGGREGATE triggers,
    # which always wins over every other signal by design ("aggregate is
    # never retrieval"). But "most urgent" is a superlative retrieval
    # genuinely answers by ranking, not a count - and nothing downstream
    # ever implements "most/fewest" as a real SQL query either
    # (router/sql.py's COUNT_SQL is defined but never called), so the only
    # effect was guaranteeing an empty result for a question retrieval could
    # have actually answered. `None` (the rules abstaining to the model) is
    # a legitimate outcome here, just not "sql" via a false-positive
    # aggregate match.
    decision = classify_rules(query)
    assert decision is None or decision.name != "sql"


@pytest.mark.parametrize("query", [
    "what was decided about the discount tier",
    "why did the merger terms change",
    "explain the confidentiality agreement dispute",
    "who said the tier was retroactive",
])
def test_content_questions_route_to_hybrid(query):
    assert classify_rules(query).name == "hybrid"


@pytest.mark.parametrize("query", [
    "what did we agree is due next week",
    "what was decided about deadlines this month",
])
def test_temporal_plus_content_routes_to_both(query):
    # The window comes from SQL, the substance from retrieval.
    assert classify_rules(query).name == "both"


def test_the_rules_abstain_rather_than_guess():
    # Abstention is a deliberate outcome that hands the query to the model, which
    # keeps router accuracy decomposable into rules-vs-model.
    assert classify_rules("calpine") is None
    assert classify_rules("") is None
    assert classify_rules("   ") is None


def test_an_unknown_route_name_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown route"):
        Route("telepathy")


# -- the router --------------------------------------------------------------

class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def json_complete(self, prompt, system="", max_tokens=2048, variant=""):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_a_rule_hit_never_calls_the_model():
    # A regex is free, deterministic and reproducible; an LLM classifier costs a
    # round-trip and can route the same question differently on two runs.
    llm = FakeLLM({"route": "hybrid", "reason": "should not be reached"})
    router = QueryRouter(llm)
    decision = router.route("what's due this week")

    assert decision.route == "sql"
    assert decision.decided_by == "rules"
    assert decision.llm_calls == 0
    assert llm.calls == 0


def test_the_model_decides_only_what_the_rules_abstained_on():
    llm = FakeLLM({"route": "hybrid", "reason": "content question"})
    router = QueryRouter(llm)
    decision = router.route("calpine")

    assert decision.route == "hybrid"
    assert decision.decided_by == "llm"
    assert decision.llm_calls == 1


def test_a_router_failure_defaults_to_both_rather_than_failing_the_query():
    # Routing a date question to search gives a wrong answer; answering both ways
    # is merely slower. Those costs are not comparable.
    router = QueryRouter(FakeLLM(RuntimeError("quota")))
    decision = router.route("calpine")

    assert decision.route == "both"
    assert decision.decided_by == "default"
    assert "router failed" in decision.reason


def test_a_nonsense_route_from_the_model_falls_back_to_both():
    router = QueryRouter(FakeLLM({"route": "telepathy"}))
    assert router.route("calpine").route == "both"


def test_the_router_can_run_with_no_model_at_all():
    router = QueryRouter(use_llm=False)
    decision = router.route("calpine")

    assert decision.route == "both"
    assert decision.decided_by == "default"
    assert decision.llm_calls == 0


def test_decisions_expose_which_arms_to_run():
    assert RouterDecision("q", "sql", "", 1.0, "rules").uses_sql
    assert not RouterDecision("q", "sql", "", 1.0, "rules").uses_retrieval
    assert RouterDecision("q", "hybrid", "", 1.0, "rules").uses_retrieval
    both = RouterDecision("q", "both", "", 1.0, "rules")
    assert both.uses_sql and both.uses_retrieval


def test_counts_report_how_much_the_rules_carried():
    router = QueryRouter(FakeLLM({"route": "hybrid", "reason": "x"}))
    router.route("what's due this week")
    router.route("what was decided about pricing")
    router.route("calpine")

    assert router.counts == {"rules": 2, "llm": 1, "default": 0}
    assert "rules=2" in router.render_counts()


# -- windows ----------------------------------------------------------------

def test_this_week_is_the_calendar_week_of_the_anchor():
    w = parse_window("what's due this week", TUE)
    assert (w.start, w.end) == (date(2001, 10, 29), date(2001, 11, 4))
    assert w.days == 7


def test_next_week_is_the_following_calendar_week():
    w = parse_window("what's due next week", TUE)
    assert (w.start, w.end) == (date(2001, 11, 5), date(2001, 11, 11))


def test_last_week_looks_backwards():
    w = parse_window("what was due last week", TUE)
    assert (w.start, w.end) == (date(2001, 10, 22), date(2001, 10, 28))


def test_yesterday_is_the_day_before_the_anchor():
    w = parse_window("what mail did I get yesterday", TUE)
    assert (w.start, w.end) == (date(2001, 10, 29), date(2001, 10, 29))


def test_month_windows_span_the_whole_month():
    w = parse_window("due this month", TUE)
    assert (w.start, w.end) == (date(2001, 10, 1), date(2001, 10, 31))
    w = parse_window("due next month", TUE)
    assert (w.start, w.end) == (date(2001, 11, 1), date(2001, 11, 30))


def test_a_quarter_window_spans_three_months():
    w = parse_window("due this quarter", TUE)
    assert (w.start, w.end) == (date(2001, 10, 1), date(2001, 12, 31))


def test_n_day_windows():
    w = parse_window("what's due in the next 10 days", TUE)
    assert (w.start, w.end) == (TUE, date(2001, 11, 9))
    w = parse_window("what came due in the past 3 days", TUE)
    assert (w.start, w.end) == (date(2001, 10, 27), TUE)


def test_an_explicit_date_becomes_a_window_ending_on_it():
    w = parse_window("what is due by November 8", TUE)
    assert (w.start, w.end) == (TUE, date(2001, 11, 8))


def test_overdue_is_everything_up_to_the_anchor():
    w = parse_window("what's overdue", TUE)
    assert w.end == TUE
    assert w.include_overdue is True


def test_overdue_is_flagged_alongside_a_window():
    w = parse_window("what's due this week or already overdue", TUE)
    assert w.include_overdue is True
    assert w.start == date(2001, 10, 29)


def test_a_question_with_no_window_returns_none():
    # Not the same as an empty window: the caller should say "no window", not
    # "nothing is due".
    assert parse_window("what was decided about pricing", TUE) is None


def test_windows_resolve_against_as_of_never_against_now():
    # The corpus is frozen in 1999-2002. Using datetime.now() would make every
    # temporal answer empty, and the failure would look like a retrieval problem.
    a = parse_window("what's due next week", date(1999, 3, 1))
    b = parse_window("what's due next week", date(2002, 6, 10))
    assert a.start.year == 1999 and b.start.year == 2002


@pytest.mark.parametrize("anchor", [
    date(2001, 10, 30),
    datetime(2001, 10, 30, 12, 0, tzinfo=timezone.utc),
    "2001-10-30",
])
def test_the_anchor_accepts_dates_datetimes_and_iso_strings(anchor):
    assert parse_window("due this week", anchor).start == date(2001, 10, 29)


def test_a_missing_or_unparseable_anchor_yields_no_window():
    assert parse_window("due this week", None) is None
    assert parse_window("due this week", "not a date") is None


def test_the_window_describes_itself_for_a_ui():
    w = parse_window("what's due next week", TUE)
    assert "2001-11-05 to 2001-11-11" in w.describe()


# -- the SQL arm ------------------------------------------------------------

def test_the_window_is_bound_as_parameters_never_interpolated():
    # A question is untrusted input from the moment it can arrive from a web form,
    # which it can as soon as phase 7 lands.
    w = parse_window("what's due next week", TUE)
    sql, params = window_sql(w, model="qwen2.5:3b")

    assert "%(start)s" in sql and "%(end)s" in sql
    assert params["start"] == date(2001, 11, 5)
    assert params["model"] == "qwen2.5:3b"
    assert "2001-11-05" not in sql


def _c(due, text="x", conf=0.5):
    return Commitment(message_id="m1", text=text, due_phrase="p", due_at=due,
                      confidence=conf)


def test_the_in_memory_filter_matches_the_sql_predicate():
    # Postgres is optional here, and the router has to be measurable without a
    # database up - so the fallback uses deliberately the same predicate.
    w = parse_window("what's due next week", TUE)
    commitments = [_c(date(2001, 11, 6)), _c(date(2001, 12, 1)),
                   _c(date(2001, 11, 11)), _c(None)]

    out = filter_commitments(commitments, w)

    assert [c.due_at for c in out] == [date(2001, 11, 6), date(2001, 11, 11)]


def test_the_window_is_inclusive_at_both_ends():
    w = parse_window("what's due next week", TUE)
    out = filter_commitments([_c(w.start), _c(w.end)], w)
    assert len(out) == 2


def test_results_are_ordered_by_date_then_confidence():
    w = parse_window("what's due this month", TUE)
    out = filter_commitments([
        _c(date(2001, 10, 20), "later", conf=0.9),
        _c(date(2001, 10, 10), "low", conf=0.2),
        _c(date(2001, 10, 10), "high", conf=0.95),
    ], w)

    assert [c.text for c in out] == ["high", "low", "later"]


# -- filtering messages by when they were *received*, not a due date --------

def _msg(day: date, subject: str = "", hour: int = 12) -> dict:
    # Noon by default - nowhere near a midnight boundary, so tests that
    # aren't specifically about timezone conversion don't accidentally
    # depend on it. `date_utc` is what `filter_messages_by_date` actually
    # reads now; `"date"` stays too since citations still display it.
    when = (datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
           if day else None)
    return {"sender": "a@x.com", "recipients": "b@x.com", "subject": subject,
           "date": day.isoformat() if day else "", "date_utc": when}


def test_filter_messages_by_date_matches_received_date_not_a_due_date():
    # The point of this arm: a message with no extracted commitment at all -
    # filter_commitments could never find this, because there is no
    # Commitment row to filter in the first place.
    w = parse_window("what did I get this week", TUE)
    messages = {
        "in-window": _msg(date(2001, 10, 30), "in window"),
        "before": _msg(date(2001, 10, 1), "too early"),
        "after": _msg(date(2001, 12, 1), "too late"),
        "no-date": _msg(None, "never dated"),
    }

    out = filter_messages_by_date(messages, w)

    assert [r["subject"] for r in out] == ["in window"]
    assert out[0]["dedup_key"] == "in-window"


def test_filter_messages_by_date_is_newest_first():
    w = parse_window("what did I get this month", TUE)
    messages = {
        "early": _msg(date(2001, 10, 5), "early"),
        "late": _msg(date(2001, 10, 25), "late"),
    }

    out = filter_messages_by_date(messages, w)

    assert [r["subject"] for r in out] == ["late", "early"]


def test_filter_messages_by_date_is_inclusive_at_both_ends():
    w = parse_window("what did I get next week", TUE)
    messages = {"start": _msg(w.start, "start"), "end": _msg(w.end, "end")}

    out = filter_messages_by_date(messages, w)

    assert {r["subject"] for r in out} == {"start", "end"}


def test_filter_messages_by_date_uses_utc_when_no_tz_is_given():
    # A message at 23:00 UTC on the 29th is still the 29th in UTC - the
    # default, with no `tz` passed, before this ever mattered to a caller
    # outside the web app.
    w = parse_window("what did I get today", date(2001, 10, 29))
    messages = {"m1": _msg(date(2001, 10, 29), "late utc", hour=23)}

    out = filter_messages_by_date(messages, w)

    assert [r["subject"] for r in out] == ["late utc"]


def test_filter_messages_by_date_buckets_by_the_given_timezone_not_utc():
    # The bug this guards against: a message sent at 23:00 UTC on the 29th
    # is already the 30th in a UTC+5 zone. A caller anchored "today" on the
    # 30th in that same zone (see chat/routes.py's `tz`) and a comparison
    # done in raw UTC would disagree about which day this message landed on
    # - exactly the gap between UTC midnight and the user's own midnight.
    from datetime import timedelta

    utc_plus_5 = timezone(timedelta(hours=5))
    w = parse_window("what did I get today", date(2001, 10, 30))
    messages = {"m1": _msg(date(2001, 10, 29), "late utc, next day at +5", hour=23)}

    without_tz = filter_messages_by_date(messages, w)
    with_tz = filter_messages_by_date(messages, w, tz=utc_plus_5)

    assert without_tz == []                  # still the 29th in UTC - outside the window
    assert [r["subject"] for r in with_tz] == ["late utc, next day at +5"]


def test_commitments_with_no_due_date_never_appear_in_a_window():
    w = parse_window("due this week", TUE)
    assert filter_commitments([_c(None)], w) == []


# -- accuracy ---------------------------------------------------------------

def _d(query, route):
    return RouterDecision(query, route, "", 1.0, "rules")


def test_router_accuracy_is_scored_against_the_eval_set_query_classes():
    decisions = [_d("q1", "sql"), _d("q2", "hybrid"), _d("q3", "hybrid")]
    classes = {"q1": "temporal", "q2": "semantic", "q3": "entity"}

    score = score_routes(decisions, classes)

    assert score["n"] == 3
    assert score["accuracy"] == 1.0
    assert score["by_class"]["temporal"]["accuracy"] == 1.0


def test_both_counts_as_correct_but_is_reported_as_over_routed():
    # The router's job is not to be minimal, it is to not miss - and the
    # over-routed column is what that policy costs.
    score = score_routes([_d("q1", "both")], {"q1": "temporal"})

    assert score["accuracy"] == 1.0
    assert score["over_routed"] == 1
    assert score["over_routed_rate"] == 1.0


def test_a_wrong_route_is_scored_wrong():
    score = score_routes([_d("q1", "hybrid")], {"q1": "temporal"})
    assert score["accuracy"] == 0.0


def test_unanswerable_controls_are_excluded_from_router_accuracy():
    # There is no right arm for a question with no answer, so counting it as a
    # hybrid win would inflate accuracy for free.
    assert CLASS_TO_ROUTE["unanswerable"] is None
    score = score_routes([_d("q1", "hybrid")], {"q1": "unanswerable"})
    assert score["n"] == 0


def test_the_table_renders_per_class_and_overall():
    score = score_routes([_d("q1", "sql"), _d("q2", "both")],
                         {"q1": "temporal", "q2": "semantic"})
    table = routes_table(score)

    assert "temporal" in table and "semantic" in table
    assert "**overall**" in table
    assert "over-routed" in table
