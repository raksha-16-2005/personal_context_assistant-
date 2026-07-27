from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.pipeline import DEFAULT_RERANK, RETRIEVE_DEPTH, collapse
from emailrag.index import rerank as RR

TEXTS = {
    "m1:0": "first chunk of message one",
    "m1:1": "second chunk of message one",
    "m2:0": "the only chunk of message two",
    "m3:0": "message three",
}
HEADERS = {
    "m1": {"sender": "a@x.com", "recipients": "b@x.com", "date": "2001-05-01",
           "subject": "Pricing"},
    "m2": {"sender": "c@x.com", "recipients": "d@x.com", "date": "2001-05-02",
           "subject": "Audit"},
    # m3 deliberately absent, to exercise a message with no metadata row.
}


def test_chunks_collapse_to_messages_keeping_best_rank():
    ranked = [("m2:0", 9.0), ("m1:1", 8.0), ("m1:0", 7.0), ("m3:0", 6.0)]

    out = collapse(ranked, 10, TEXTS, HEADERS)

    assert [m.message_id for m in out] == ["m2", "m1", "m3"]
    assert out[1].score == 8.0            # m1's best chunk, not its last
    assert out[1].rank == 2


def test_all_retrieved_chunks_of_a_message_are_concatenated_in_rank_order():
    # A citation should show everything retrieval found in that message, not an
    # arbitrary one of its fragments.
    ranked = [("m1:1", 9.0), ("m1:0", 8.0)]

    out = collapse(ranked, 10, TEXTS, HEADERS)

    assert len(out) == 1
    assert out[0].chunk_ids == ["m1:1", "m1:0"]
    assert out[0].text == "second chunk of message one\nfirst chunk of message one"


def test_identical_chunk_text_is_not_shown_twice():
    # Overlapping chunkings (fixed_512_ov64) can retrieve two chunks whose text
    # is the same; concatenating them would show a reader the same paragraph
    # twice in one excerpt.
    out = collapse([("m1:0", 9.0), ("m1:1", 8.0)], 10,
                   {"m1:0": "same text", "m1:1": "same text"}, HEADERS)

    assert out[0].text == "same text"
    assert out[0].chunk_ids == ["m1:0", "m1:1"]      # both still recorded


def test_top_n_counts_messages_not_chunks():
    # The caller asked for two things to read, not two fragments of one thing.
    ranked = [("m1:0", 9.0), ("m1:1", 8.5), ("m2:0", 8.0), ("m3:0", 7.0)]

    out = collapse(ranked, 2, TEXTS, HEADERS)

    assert [m.message_id for m in out] == ["m1", "m2"]


def test_later_chunks_of_an_included_message_survive_the_top_n_cut():
    # m1 is already in the list when its second chunk appears after the cut;
    # dropping it would render a partial excerpt of a message retrieved twice.
    ranked = [("m1:0", 9.0), ("m2:0", 8.0), ("m3:0", 7.0), ("m1:1", 6.0)]

    out = collapse(ranked, 2, TEXTS, HEADERS)

    assert [m.message_id for m in out] == ["m1", "m2"]
    assert out[0].chunk_ids == ["m1:0", "m1:1"]
    assert "second chunk" in out[0].text


def test_metadata_is_attached_when_present():
    out = collapse([("m1:0", 1.0)], 5, TEXTS, HEADERS)

    assert out[0].sender == "a@x.com"
    assert out[0].date == "2001-05-01"
    assert out[0].subject == "Pricing"


def test_a_message_without_metadata_still_renders():
    # Better an untitled result than a crash: the index and the parquet can drift.
    out = collapse([("m3:0", 1.0)], 5, TEXTS, HEADERS)

    assert out[0].message_id == "m3"
    assert out[0].sender == "" and out[0].subject == ""
    assert out[0].text == "message three"


def test_a_chunk_with_no_text_yields_an_empty_excerpt_not_an_error():
    out = collapse([("ghost:0", 1.0)], 5, TEXTS, HEADERS)

    assert out[0].message_id == "ghost" and out[0].text == ""


def test_empty_ranking_collapses_to_nothing():
    assert collapse([], 10, TEXTS, HEADERS) == []


# -- configuration -----------------------------------------------------------

def test_the_default_rerank_arm_is_one_that_fits_the_latency_budget():
    # bench_rerank_budget measured L2@20/t192 at 158 ms; every other arm is over
    # the 200 ms shipped-path budget on this CPU.
    assert DEFAULT_RERANK in RR.SPECS
    spec = RR.SPECS[DEFAULT_RERANK]
    assert spec is not None
    assert spec.top_k == 20 and spec.max_length == 192


def test_retrieve_depth_exceeds_what_is_displayed():
    # Several chunks routinely belong to one message, so 60 chunks can be far
    # fewer than 60 messages.
    assert RETRIEVE_DEPTH >= 100


# -- the default temporal anchor ---------------------------------------------

def test_the_corpus_end_anchor_ignores_garbage_dates():
    # Enron's headers contain junk: the real 50k sample spans 1980-01-01 to
    # 2044-01-04 against a true span of 1999-2002. Anchoring "what's due next week"
    # on max() would put the window in 2044, every temporal query would return
    # nothing, and the emptiness would read as a retrieval failure.
    from datetime import datetime, timezone

    from emailrag.pipeline import _corpus_end_date

    real = [datetime(2001, 5, i % 28 + 1, tzinfo=timezone.utc) for i in range(200)]
    junk = [datetime(2044, 1, 4, tzinfo=timezone.utc),
            datetime(1980, 1, 1, tzinfo=timezone.utc)]

    anchor = _corpus_end_date(real + junk)

    assert anchor.startswith("2001")


def test_the_anchor_is_empty_when_nothing_is_dated():
    from emailrag.pipeline import _corpus_end_date
    assert _corpus_end_date([]) == ""
