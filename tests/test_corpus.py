from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.corpus import filters
from emailrag.corpus.enron import _normalize_subject, _strip_quoted, parse_file
from emailrag.corpus.sample import stratified_thread_sample
from emailrag.corpus.threads import assign_threads

UTC = timezone.utc

RAW = b"""Message-ID: <123.456.JavaMail.evans@thyme>
Date: Tue, 5 Dec 2000 09:14:00 -0800 (PST)
From: sara.shackleton@enron.com
To: mark.taylor@enron.com, Rick.Buy@enron.com
Subject: RE: Fwd: pricing model
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
X-Folder: \\Sara_Shackleton_Dec2000\\Notes Folders\\Sent

Mark - legal signed off on 4.2 this morning. We still need Rick's number.

-----Original Message-----
From: Taylor, Mark
Sent: Monday, December 04, 2000 4:02 PM
To: Shackleton, Sara
Subject: pricing model

> what did we land on for the discount tier
"""


def _write(tmp_path: Path, name: str, raw: bytes) -> Path:
    p = tmp_path / "maildir" / "shackleton-s" / "sent"
    p.mkdir(parents=True, exist_ok=True)
    f = p / name
    f.write_bytes(raw)
    return f


def test_parse_extracts_normalized_fields(tmp_path):
    f = _write(tmp_path, "1.", RAW)
    msg = parse_file(f, tmp_path / "maildir")

    assert msg is not None
    assert msg.message_id == "<123.456.JavaMail.evans@thyme>"
    assert msg.sender == "sara.shackleton@enron.com"
    # Addresses are lowercased and ';'-joined, order preserved.
    assert msg.recipients == "mark.taylor@enron.com;rick.buy@enron.com"
    assert msg.date_utc == datetime(2000, 12, 5, 17, 14, tzinfo=UTC)
    assert msg.owner == "shackleton-s"
    assert msg.folder == "sent"


def test_quoted_history_is_stripped_from_body_new():
    msg_body = RAW.decode().split("\n\n", 1)[1]
    new = _strip_quoted(msg_body)

    assert "legal signed off on 4.2" in new
    # Everything from the Outlook banner onward belongs to the earlier message.
    assert "Original Message" not in new
    assert "discount tier" not in new


def test_subject_normalization_strips_stacked_prefixes():
    assert _normalize_subject("RE: Fwd: pricing model") == "pricing model"
    assert _normalize_subject("Re: RE: re: Audit") == "audit"
    assert _normalize_subject("pricing model") == "pricing model"


def test_eight_bit_header_bytes_do_not_crash_the_parse(tmp_path):
    # Raw non-ASCII bytes in a header make compat32 surrogate-escape the value
    # and hand back an `email.header.Header`, which has no string methods.
    # Enron has enough of these to abort a full parse partway through.
    raw = (b"Message-ID: <8bit@thyme>\n"
           b"From: Jos\xe9 Ramirez <jose@enron.com>\n"
           b"To: mark.taylor@enron.com\n"
           b"Subject: caf\xe9 meeting notes\n"
           b"Date: Tue, 5 Dec 2000 09:14:00 -0800 (PST)\n\n"
           b"Body text long enough to survive the bulk filter threshold.\n")
    msg = parse_file(_write(tmp_path, "8bit.", raw), tmp_path / "maildir")

    assert msg is not None
    assert msg.sender == "jose@enron.com"
    assert "meeting notes" in msg.subject


def test_rfc2047_encoded_subject_is_decoded_not_stringified(tmp_path):
    # A real Gmail message hit this: compat32 hands back an
    # `email.header.Header` for an encoded-word subject, and `str()` on that
    # object returns the encoded form verbatim (`=?Windows-1252?Q?...?=`)
    # rather than decoding it - the subject that reached citations was
    # literally that gibberish until _hdr called decode_header.
    raw = (b"Message-ID: <encoded@thyme>\n"
          b"From: alerts@example.com\n"
          b"To: mark.taylor@enron.com\n"
          b"Subject: =?Windows-1252?Q?Don=92t_Miss_Out!_Deadline_March_23!?=\n"
          b"Date: Tue, 5 Dec 2000 09:14:00 -0800 (PST)\n\n"
          b"Body text long enough to survive the bulk filter threshold.\n")
    msg = parse_file(_write(tmp_path, "encoded.", raw), tmp_path / "maildir")

    assert msg is not None
    assert "=?" not in msg.subject
    assert "Don" in msg.subject and "Miss Out" in msg.subject
    assert "Deadline March 23" in msg.subject


def test_dedup_is_content_based_not_message_id(tmp_path):
    # The Enron export stamped a distinct Message-ID on every copy of a
    # message, so ID-based dedup removes nothing. Identical content under
    # different ids must still collapse to one row.
    a = parse_file(_write(tmp_path, "a.", RAW), tmp_path / "maildir")
    b = parse_file(
        _write(tmp_path, "b.", RAW.replace(b"<123.456.JavaMail.evans@thyme>",
                                           b"<999.888.JavaMail.evans@thyme>")),
        tmp_path / "maildir")

    assert a.message_id != b.message_id
    assert a.dedup_key.startswith("sha256:")
    assert a.dedup_key == b.dedup_key


def test_dedup_keeps_different_recipients_apart(tmp_path):
    # Two genuinely separate sends of the same text to different people are
    # not duplicates - entity-scoped queries depend on telling them apart.
    a = parse_file(_write(tmp_path, "a.", RAW), tmp_path / "maildir")
    b = parse_file(
        _write(tmp_path, "b.", RAW.replace(b"To: mark.taylor@enron.com, Rick.Buy@enron.com",
                                           b"To: jeff.skilling@enron.com")),
        tmp_path / "maildir")

    assert a.dedup_key != b.dedup_key


def test_missing_message_id_still_parses(tmp_path):
    raw = RAW.replace(b"Message-ID: <123.456.JavaMail.evans@thyme>\n", b"")
    msg = parse_file(_write(tmp_path, "a.", raw), tmp_path / "maildir")

    assert msg is not None
    assert msg.message_id == ""
    assert msg.dedup_key.startswith("sha256:")


# --- filters ---------------------------------------------------------------

def _msg(**kw) -> dict:
    base = {"sender": "a@enron.com", "subject": "contract", "body_new": "x" * 100,
            "has_list_unsubscribe": False}
    base.update(kw)
    return base


def test_bulk_filter_catches_machine_mail_and_keeps_real_mail():
    assert filters.is_bulk(_msg(has_list_unsubscribe=True))
    assert filters.is_bulk(_msg(sender="no-reply@example.com"))
    assert filters.is_bulk(_msg(sender="enron.announcements@enron.com"))
    assert filters.is_bulk(_msg(subject="Out of Office AutoReply"))
    assert filters.is_bulk(_msg(subject="Re: Undeliverable: meeting"))
    assert filters.is_bulk(_msg(body_new="ok"))          # no retrievable content

    assert not filters.is_bulk(_msg())
    # 'notification' must match as a token, not inside an ordinary name.
    assert not filters.is_bulk(_msg(sender="john.reply@enron.com"))


def test_dedup_and_filter_drops_repeats_and_bulk_in_one_pass():
    messages = [
        _msg(dedup_key="k1"),
        _msg(dedup_key="k1"),                          # exact repeat
        _msg(dedup_key="k2", has_list_unsubscribe=True),  # bulk
        _msg(dedup_key="k3"),
    ]
    kept, stats = filters.dedup_and_filter(messages)

    assert [m["dedup_key"] for m in kept] == ["k1", "k3"]
    assert stats == {"n_parsed": 4, "n_dupe": 1, "n_bulk": 1, "n_kept": 2}


def test_dedup_and_filter_keep_bulk_disables_the_rules_filter():
    messages = [_msg(dedup_key="k1", has_list_unsubscribe=True)]
    kept, stats = filters.dedup_and_filter(messages, keep_bulk=True)

    assert [m["dedup_key"] for m in kept] == ["k1"]
    assert stats["n_bulk"] == 0


def test_dedup_and_filter_seen_set_persists_across_calls():
    # This is what lets the per-user Gmail ingestion worker dedup across
    # incremental sync batches without holding every previously-fetched
    # message - only its dedup keys, in `seen`.
    seen: set[str] = set()
    first, _ = filters.dedup_and_filter([_msg(dedup_key="k1")], seen=seen)
    second, stats = filters.dedup_and_filter([_msg(dedup_key="k1")], seen=seen)

    assert len(first) == 1
    assert len(second) == 0
    assert stats["n_dupe"] == 1


# --- threading -------------------------------------------------------------

def _row(key, subject, sender, to, day, **kw) -> dict:
    row = {
        "dedup_key": key, "message_id": f"<{key}@enron>", "in_reply_to": "",
        "references": "", "subject_norm": subject, "sender": sender,
        "recipients": to, "cc": "", "date_utc": datetime(2000, 12, 1, tzinfo=UTC) + timedelta(days=day),
    }
    row.update(kw)
    return row


def test_threading_links_by_reply_headers():
    rows = [
        _row("m1", "pricing model", "a@e.com", "b@e.com", 0),
        _row("m2", "pricing model", "b@e.com", "a@e.com", 1, in_reply_to="<m1@enron>"),
    ]
    tids, stats = assign_threads(rows)

    assert tids[0] == tids[1]
    assert stats["linked_by_header"] == 1
    assert stats["multi_message_threads"] == 1


def test_threading_falls_back_to_subject_plus_participant_overlap():
    # No reply headers - the JavaMail export case that makes pass 2 necessary.
    rows = [
        _row("m1", "quarterly audit", "a@e.com", "b@e.com", 0),
        _row("m2", "quarterly audit", "b@e.com", "a@e.com", 2),
    ]
    tids, stats = assign_threads(rows)

    assert tids[0] == tids[1]
    assert stats["linked_by_heuristic"] == 1


def test_threading_does_not_merge_on_subject_alone():
    # Same subject, disjoint participants -> must stay separate. This is the
    # case that makes a subject-only threader unusable on a corpus where
    # "Re: meeting" recurs for years.
    rows = [
        _row("m1", "meeting", "a@e.com", "b@e.com", 0),
        _row("m2", "meeting", "x@e.com", "y@e.com", 1),
    ]
    tids, _ = assign_threads(rows)
    assert tids[0] != tids[1]


def test_threading_respects_the_time_window():
    rows = [
        _row("m1", "budget review", "a@e.com", "b@e.com", 0),
        _row("m2", "budget review", "a@e.com", "b@e.com", 200),   # > 90 days
    ]
    tids, _ = assign_threads(rows)
    assert tids[0] != tids[1]


def test_thread_id_is_the_earliest_member_and_order_independent():
    rows = [
        _row("m2", "pricing", "b@e.com", "a@e.com", 5, in_reply_to="<m1@enron>"),
        _row("m1", "pricing", "a@e.com", "b@e.com", 0),
    ]
    tids, _ = assign_threads(rows)
    assert set(tids) == {"m1"}   # earliest message names the thread


# --- sampling --------------------------------------------------------------

def test_sample_keeps_threads_intact_and_is_deterministic():
    rows = []
    for t in range(60):
        for i in range(3):
            r = _row(f"t{t}m{i}", f"subject {t}", "a@e.com", "b@e.com", t)
            r["thread_id"] = f"t{t}"
            rows.append(r)

    got, stats = stratified_thread_sample(rows, target_messages=30, seed=7)
    again, _ = stratified_thread_sample(rows, target_messages=30, seed=7)

    assert [m["dedup_key"] for m in got] == [m["dedup_key"] for m in again]
    # No thread may be partially present.
    by_thread: dict[str, int] = {}
    for m in got:
        by_thread[m["thread_id"]] = by_thread.get(m["thread_id"], 0) + 1
    assert set(by_thread.values()) == {3}
    assert stats["sampled_messages"] == len(got)


def test_sample_returns_everything_when_target_exceeds_corpus():
    rows = [dict(_row("a", "s", "a@e.com", "b@e.com", 0), thread_id="t1")]
    got, _ = stratified_thread_sample(rows, target_messages=999, seed=1)
    assert len(got) == 1


# --- corpus-level bulk detection -------------------------------------------

def test_recurring_sender_subject_pairs_are_detected():
    # 623 identical subjects from one automated sender is the pattern that
    # wrecked thread reconstruction; a human thread never looks like this.
    pairs = ([("crawler@enron.com", "hourahead failure")] * 30
             + [("sara@enron.com", "pricing model")] * 4)
    blasts = filters.find_recurring_blasts(pairs, threshold=25)

    assert ("crawler@enron.com", "hourahead failure") in blasts
    assert ("sara@enron.com", "pricing model") not in blasts


def test_recurring_detection_ignores_empty_subjects():
    blasts = filters.find_recurring_blasts([("a@enron.com", "")] * 100, threshold=25)
    assert blasts == set()


def test_oversized_subject_buckets_are_not_threaded():
    # A petition: many distinct senders, one shared recipient, one subject.
    # Consecutive pairs all overlap on the recipient, so without the cap the
    # union-find chains every one of them into a single thread.
    rows = [_row(f"p{i}", "demand ken lay donate proceeds", f"s{i}@ext.com",
                 "ken.lay@enron.com", i % 60) for i in range(120)]
    tids, stats = assign_threads(rows)

    assert stats["oversized_buckets_skipped"] == 1
    assert stats["largest_thread"] == 1          # all left as singletons
    assert len(set(tids)) == 120


def test_normal_sized_buckets_still_thread():
    rows = [_row(f"m{i}", "quarterly audit", "a@e.com", "b@e.com", i) for i in range(5)]
    _tids, stats = assign_threads(rows)
    assert stats["oversized_buckets_skipped"] == 0
    assert stats["largest_thread"] == 5
