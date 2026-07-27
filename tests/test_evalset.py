from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation.evalset import (
    EvalQuery, ValidationError, load, stratification, validate,
)


def _q(qid="q001", cls="semantic", ids=("m1",), **kw) -> EvalQuery:
    return EvalQuery(query_id=qid, query=kw.pop("query", f"query text {qid}"),
                     query_class=cls, relevant_message_ids=list(ids),
                     verified=kw.pop("verified", True), **kw)


def test_answerable_query_without_labels_is_an_error():
    report = validate([_q(ids=[])])
    assert not report.ok
    assert "no relevant_message_ids" in report.errors[0]


def test_unanswerable_control_with_labels_is_an_error():
    report = validate([_q(cls="unanswerable", ids=["m1"])])
    assert not report.ok
    assert "must have none" in report.errors[0]


def test_unanswerable_control_without_labels_is_valid():
    report = validate([_q(cls="unanswerable", ids=[])])
    assert not report.errors


def test_labels_outside_the_corpus_are_caught():
    report = validate([_q(ids=["m1", "ghost"])], corpus_ids={"m1"})
    assert not report.ok
    assert "absent from the corpus" in report.errors[0]
    assert "ghost" in report.errors[0]


def test_labels_inside_the_corpus_pass():
    assert validate([_q(ids=["m1"])], corpus_ids={"m1", "m2"}).ok


def test_duplicate_query_ids_are_caught():
    report = validate([_q(qid="q001"), _q(qid="q001")])
    assert any("duplicate query_id" in e for e in report.errors)


def test_duplicate_message_ids_within_a_query_are_caught():
    report = validate([_q(ids=["m1", "m1"])])
    assert any("duplicate ids" in e for e in report.errors)


def test_unknown_query_class_is_caught():
    report = validate([_q(cls="freeform")])
    assert any("unknown query_class" in e for e in report.errors)


def test_near_duplicate_queries_warn_but_do_not_fail():
    report = validate([
        _q(qid="q001", query="What did we decide about pricing?"),
        _q(qid="q002", query="what did we  DECIDE about  pricing?"),
    ])
    assert report.ok
    assert any("near-duplicate" in w for w in report.warnings)


def test_unverified_queries_warn():
    report = validate([_q(verified=False)])
    assert report.ok
    assert any("not hand-verified" in w for w in report.warnings)


def test_stratification_shortfall_warns():
    report = validate([_q()])
    assert any("plan targets 35" in w for w in report.warnings)


def test_roundtrip_from_jsonl(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "# comment lines and blanks are skipped\n\n"
        + json.dumps({"query_id": "q001", "query": "pricing decision",
                      "query_class": "semantic", "relevant_message_ids": ["m1"],
                      "verified": True}) + "\n"
        + json.dumps({"query_id": "q002", "query": "who won the 2030 world cup",
                      "query_class": "unanswerable", "relevant_message_ids": []}) + "\n"
    )
    queries = load(path)

    assert [q.query_id for q in queries] == ["q001", "q002"]
    assert queries[0].answerable and not queries[1].answerable
    assert stratification(queries) == {"semantic": 1, "temporal": 0,
                                       "entity": 0, "unanswerable": 1}


def test_malformed_json_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query_id": "q001"\n')
    with pytest.raises(ValidationError, match="bad.jsonl:1"):
        load(path)


def test_missing_field_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"query_id": "q001", "query": "x"}) + "\n")
    with pytest.raises(ValidationError, match="missing field"):
        load(path)


# --- temporal anchoring ----------------------------------------------------

def test_temporal_query_without_as_of_warns():
    # "what is due next week" over a frozen 2001 archive resolves to nothing;
    # scoring it as a retrieval failure would be wrong.
    report = validate([_q(cls="temporal", ids=["m1"])])
    assert report.ok                                   # warning, not an error
    assert any("without as_of" in w for w in report.warnings)


def test_temporal_query_with_as_of_is_clean():
    report = validate([_q(cls="temporal", ids=["m1"], as_of="2001-03-12")])
    assert not any("as_of" in w for w in report.warnings)


def test_malformed_as_of_is_an_error():
    report = validate([_q(cls="temporal", ids=["m1"], as_of="March 12 2001")])
    assert not report.ok
    assert any("not YYYY-MM-DD" in e for e in report.errors)


def test_non_temporal_classes_need_no_as_of():
    report = validate([_q(cls="semantic", ids=["m1"])])
    assert not any("as_of" in w for w in report.warnings)


def test_as_of_survives_the_jsonl_roundtrip(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({
        "query_id": "q001", "query": "what was due that week",
        "query_class": "temporal", "relevant_message_ids": ["m1"],
        "verified": True, "as_of": "2001-03-12"}) + "\n")

    assert load(path)[0].as_of == "2001-03-12"
