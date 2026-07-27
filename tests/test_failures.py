from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation import failures as F
from emailrag.evaluation.evalset import EvalQuery


def _q(qid, text, cls, rel, notes="", as_of=""):
    return EvalQuery(qid, text, cls, list(rel), notes=notes, verified=True, as_of=as_of)


def _record(qid, found_at=None, chunk_ranks=()):
    return {"query_id": qid, "found_at": found_at, "n_relevant": 1,
            "relevant_chunk_ranks": list(chunk_ranks)}


def _classify(q, record, chunk_ranks=None, texts=None, k=20):
    return F.classify([record], {q.query_id: q},
                      chunk_ranks or {}, texts or {}, k=k)


# -- what counts as a miss --------------------------------------------------

def test_a_hit_inside_k_is_not_a_miss():
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=3))

    assert report.misses == []
    assert report.n_answerable == 1


def test_a_hit_outside_k_is_a_miss():
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=25))

    assert len(report.misses) == 1


def test_unanswerable_controls_are_not_counted_at_all():
    # There is no correct document to retrieve, so there is nothing to explain.
    q = _q("q1", "who won the 2030 world cup", "unanswerable", [])
    report = _classify(q, _record("q1", found_at=None))

    assert report.misses == [] and report.n_answerable == 0


# -- category precedence ----------------------------------------------------

def test_only_a_human_can_mark_a_label_bad():
    # Auto-assigning this would let the system grade its own exam: any query it
    # failed could be dismissed as mislabelled.
    q = _q("q1", "pricing", "semantic", ["m1"],
           notes="bad_label: this thread is about a different contract")
    report = _classify(q, _record("q1", found_at=None))

    assert report.misses[0].category == "bad_label"


def test_bad_label_wins_over_every_other_signal():
    q = _q("q1", "pricing", "temporal", ["m1", "m2"], notes="bad_label: wrong",
           as_of="2001-10-30")
    report = _classify(q, _record("q1", found_at=None, chunk_ranks=[("m1:0", 3)]))

    assert report.misses[0].category == "bad_label"


def test_a_chunk_of_the_right_message_ranking_well_is_a_chunking_failure():
    # The message was found; the chunk that answers was not. That is dimension
    # 1's problem, not the retriever's.
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=None, chunk_ranks=[("m1:4", 7)]),
                       chunk_ranks={"q1": [("m1:4", 7)]})

    miss = report.misses[0]
    assert miss.category == "chunk_boundary"
    assert "rank 7" in miss.detail


def test_a_chunk_ranking_outside_k_is_not_a_chunk_boundary_failure():
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=None),
                       chunk_ranks={"q1": [("m1:4", 150)]})

    assert report.misses[0].category != "chunk_boundary"


def test_temporal_queries_are_attributed_to_time_not_vocabulary():
    # Nothing in dense or sparse retrieval models a date; this is the router's
    # job, and filing it as vocabulary would point the fix at expansion.
    q = _q("q1", "what is due next week", "temporal", ["m1"], as_of="2001-10-30")
    report = _classify(q, _record("q1", found_at=None))

    assert report.misses[0].category == "temporal"
    assert "2001-10-30" in report.misses[0].detail


def test_several_labelled_messages_is_multi_hop():
    q = _q("q1", "what did legal say before the committee met", "semantic",
           ["m1", "m2", "m3"])
    report = _classify(q, _record("q1", found_at=None))

    miss = report.misses[0]
    assert miss.category == "multi_hop"
    assert miss.n_relevant == 3


def test_temporal_beats_multi_hop():
    q = _q("q1", "what was due", "temporal", ["m1", "m2"], as_of="2001-01-01")
    report = _classify(q, _record("q1", found_at=None))

    assert report.misses[0].category == "temporal"


# -- the residual split -----------------------------------------------------

def test_low_term_overlap_is_a_vocabulary_mismatch():
    q = _q("q1", "who owns the calpine renewal", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=None),
                       texts={"m1": "picked it up when sara moved desks"})

    miss = report.misses[0]
    assert miss.category == "vocabulary"
    assert miss.term_overlap < F.VOCAB_OVERLAP_THRESHOLD


def test_high_term_overlap_is_a_ranking_failure_not_a_vocabulary_one():
    # The terms were already there. Query expansion cannot fix this; reordering
    # might. Filing it as vocabulary would point at the wrong fix.
    q = _q("q1", "confidentiality agreement changes", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=70),
                       texts={"m1": "changes needed for the confidentiality "
                                    "agreement before friday"})

    miss = report.misses[0]
    assert miss.category == "ranking"
    assert miss.term_overlap == 1.0
    assert "reordering" in miss.detail


def test_missing_text_does_not_silently_claim_a_vocabulary_mismatch():
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=None), texts={})

    assert "unmeasured" in report.misses[0].detail


# -- overlap measurement ----------------------------------------------------

def test_stopwords_do_not_shrink_the_denominator():
    # "what did we decide about the tier" has two content terms, decide and
    # tier. If the six stopwords counted, a document matching both would score
    # 0.29 and look like a vocabulary mismatch.
    assert F.overlap("what did we decide about the tier", "decide tier") == 1.0
    assert F.overlap("what did we decide about the tier", "the tier") == 0.5


def test_stopwords_in_the_document_cannot_manufacture_overlap():
    # Otherwise every query overlaps every email through "the" and "we".
    assert F.overlap("calpine renewal", "we sent it to the desk and they are on it") == 0.0


def test_overlap_is_zero_for_disjoint_text():
    assert F.overlap("calpine renewal", "cafeteria menu tuesday") == 0.0


def test_overlap_of_an_all_stopword_query_is_zero_not_a_crash():
    assert F.overlap("what is it about", "anything") == 0.0


# -- chunk rank extraction --------------------------------------------------

def test_chunk_ranks_finds_every_chunk_of_a_labelled_message():
    ranked = [("m9:0", 1.0), ("m1:3", 0.9), ("m2:0", 0.8), ("m1:0", 0.7)]

    out = F.chunk_ranks_from_run(ranked, {"m1"})

    assert out == [("m1:3", 2), ("m1:0", 4)]


def test_chunk_ranks_are_positions_before_the_message_collapse():
    # After collapsing, m1's second chunk is gone and the chunk-boundary signal
    # with it.
    ranked = [("m1:5", 1.0)] + [(f"m{i}:0", 0.5) for i in range(2, 30)]

    assert F.chunk_ranks_from_run(ranked, {"m1"}) == [("m1:5", 1)]


# -- report -----------------------------------------------------------------

def test_report_renders_counts_and_shares():
    queries = {
        "q1": _q("q1", "a temporal one", "temporal", ["m1"], as_of="2001-01-01"),
        "q2": _q("q2", "a multi hop one", "semantic", ["m1", "m2"]),
        "q3": _q("q3", "another multi hop", "semantic", ["m3", "m4"]),
    }
    records = [_record(qid, found_at=None) for qid in queries]

    report = F.classify(records, queries, {}, {})
    table = report.render()

    assert report.counts == {"temporal": 1, "multi_hop": 2}
    assert "| temporal | 1 | 33% |" in table
    assert "| multi_hop | 2 | 67% |" in table
    assert "3 of 3 answerable queries miss at recall@20" in table


def test_report_is_empty_when_nothing_misses():
    q = _q("q1", "pricing", "semantic", ["m1"])
    report = _classify(q, _record("q1", found_at=1))

    assert report.counts == {}
    assert "0 of 1" in report.render()
