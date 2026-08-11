from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import Settings
from app.ingestion.worker import (bulk_messages_path, index_dir, messages_path,
                                  schedule_due_syncs, sync_state_path, sync_user)
from app.tokens.store import DbTokenStore
from emailrag.corpus import gmail as G

RAW_1 = (b"From: alice@example.com\r\nTo: bob@example.com\r\n"
        b"Subject: Project kickoff\r\nDate: Mon, 5 Jan 2026 09:00:00 -0000\r\n"
        b"Message-ID: <m1@example.com>\r\n\r\n"
        b"Let's kick off the project next week. I'll send the agenda by Friday.\r\n")
RAW_2 = (b"From: bob@example.com\r\nTo: alice@example.com\r\n"
        b"Subject: Re: Project kickoff\r\nDate: Mon, 5 Jan 2026 10:00:00 -0000\r\n"
        b"Message-ID: <m2@example.com>\r\nIn-Reply-To: <m1@example.com>\r\n\r\n"
        b"Sounds good, looking forward to the agenda.\r\n")
RAW_3 = (b"From: carol@example.com\r\nTo: alice@example.com\r\n"
        b"Subject: Budget review\r\nDate: Tue, 6 Jan 2026 09:00:00 -0000\r\n"
        b"Message-ID: <m3@example.com>\r\n\r\n"
        b"Can we review the Q1 budget numbers this week?\r\n")
RAW_BULK = (b"From: updates@newsletter.example.com\r\nTo: alice@example.com\r\n"
           b"Subject: Weekly newsletter\r\nDate: Tue, 6 Jan 2026 11:00:00 -0000\r\n"
           b"Message-ID: <m4@example.com>\r\n"
           b"List-Unsubscribe: <mailto:unsub@newsletter.example.com>\r\n\r\n"
           b"Check out this week's top stories from our newsletter.\r\n")

RAW_BY_ID = {"m1": RAW_1, "m2": RAW_2, "m3": RAW_3}


def _fake_list_message_ids(self, query="", max_messages=0, include_spam_trash=False):
    return iter(RAW_BY_ID.keys())


def _fake_raw_message(self, message_id):
    return RAW_BY_ID[message_id]


def _fake_current_history_id(self):
    return "1000"


@pytest.fixture()
def gmail_class_mocked(monkeypatch):
    # Class-level, not instance-level: sync_user() constructs its own
    # GmailClient internally, so there is no instance handle to patch before
    # it exists - matching test_gmail.py's own instance-patching style, one
    # level up.
    monkeypatch.setattr(G.GmailClient, "list_message_ids", _fake_list_message_ids)
    monkeypatch.setattr(G.GmailClient, "raw_message", _fake_raw_message)
    monkeypatch.setattr(G.GmailClient, "current_history_id", _fake_current_history_id)


def _settings(tmp_path) -> Settings:
    return Settings(
        database_url="postgresql:///emailrag",
        master_key=Fernet.generate_key().decode(),
        session_secret="test-session-secret",
        gmail_client_id="test-client-id",
        gmail_client_secret="test-client-secret",
        oauth_redirect_base_url="http://localhost:8000",
        user_index_root=tmp_path,
        shipped_chunking="thread_aware",
        shipped_model="sentence-transformers/all-MiniLM-L6-v2",
        shipped_rerank="L2@20/t192",
    )


def _seed_tokens(db_conn, user_id, master_key):
    DbTokenStore(conn=db_conn, user_id=user_id, master_key=master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()


@pytest.mark.slow
def test_a_first_sync_writes_messages_and_a_real_index(
        db_conn, test_user, tmp_path, gmail_class_mocked):
    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    summary = sync_user(db_conn, settings, test_user)

    assert summary["new_messages"] == 3
    assert summary["total_messages"] == 3
    # thread_aware chunking merges the two-message kickoff thread into one
    # chunk when it's short enough to fit, so 2 chunks (not 3) is correct.
    assert summary["n_chunks"] >= 2

    msg_path = messages_path(tmp_path, test_user)
    assert msg_path.exists()
    import pyarrow.parquet as pq
    rows = pq.read_table(msg_path).to_pylist()
    assert {r["sender"] for r in rows} == {
        "alice@example.com", "bob@example.com", "carol@example.com"}
    # Threading: m1/m2 share a thread (reply headers), m3 is its own.
    by_key = {r["subject"]: r["thread_id"] for r in rows}
    assert by_key["Project kickoff"] == by_key["Re: Project kickoff"]
    assert by_key["Budget review"] != by_key["Project kickoff"]

    idx = index_dir(tmp_path, test_user)
    assert (idx / "config.json").exists()
    assert (idx / "bm25").exists()
    assert (idx / "dense").exists()
    assert (idx / "chunks.jsonl").exists()

    status = db_conn.execute(
        "SELECT status, messages_seen FROM sync_state WHERE user_id = %s",
        (test_user,)).fetchone()
    assert status[0] == "ready"
    assert status[1] == 3


@pytest.mark.slow
def test_bulk_mail_is_tracked_separately_not_indexed(
        db_conn, test_user, tmp_path, gmail_class_mocked, monkeypatch):
    """corpus.filters.is_bulk already drops newsletter mail from the
    searchable corpus (see filters.py) - this only checks the *new* half:
    that the dropped message still gets recorded, lightly, in its own file
    rather than vanishing without a trace (see _new_bulk_rows/pipeline.py's
    `_message_date_arm`, which is what turns this into an honest "how many"
    count instead of a silent undercount).
    """
    by_id = {**RAW_BY_ID, "m4": RAW_BULK}
    monkeypatch.setattr(G.GmailClient, "list_message_ids",
                        lambda self, query="", max_messages=0,
                        include_spam_trash=False: iter(by_id.keys()))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: by_id[message_id])

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    summary = sync_user(db_conn, settings, test_user)

    assert summary["new_messages"] == 3            # the newsletter never counts as "kept"
    assert summary["new_bulk_messages"] == 1

    import pyarrow.parquet as pq
    kept_senders = {r["sender"] for r in pq.read_table(messages_path(tmp_path, test_user)).to_pylist()}
    assert "updates@newsletter.example.com" not in kept_senders

    bulk_path = bulk_messages_path(tmp_path, test_user)
    assert bulk_path.exists()
    bulk_rows = pq.read_table(bulk_path).to_pylist()
    assert len(bulk_rows) == 1
    assert bulk_rows[0]["sender"] == "updates@newsletter.example.com"
    # Header-only, on purpose - no subject/body column exists to check, which
    # is the point: this file can never become a source of searchable content.
    assert set(bulk_rows[0]) == {"dedup_key", "date_utc", "sender"}


@pytest.mark.slow
def test_a_second_sync_does_not_re_record_the_same_bulk_message(
        db_conn, test_user, tmp_path, gmail_class_mocked, monkeypatch):
    by_id = {**RAW_BY_ID, "m4": RAW_BULK}
    monkeypatch.setattr(G.GmailClient, "list_message_ids",
                        lambda self, query="", max_messages=0,
                        include_spam_trash=False: iter(by_id.keys()))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: by_id[message_id])

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    sync_user(db_conn, settings, test_user)                       # first sync

    monkeypatch.setattr(G.GmailClient, "history_since", lambda self, h: iter([]))
    summary = sync_user(db_conn, settings, test_user)              # second: incremental, nothing new

    assert summary["new_bulk_messages"] == 0
    import pyarrow.parquet as pq
    assert len(pq.read_table(bulk_messages_path(tmp_path, test_user)).to_pylist()) == 1


@pytest.mark.slow
def test_a_second_sync_merges_rather_than_duplicates(
        db_conn, test_user, tmp_path, gmail_class_mocked, monkeypatch):
    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    sync_user(db_conn, settings, test_user)          # first sync: m1, m2, m3

    # Second sync: incremental path (history_id now stored from the first
    # run) returns nothing new - re-running must not duplicate the corpus.
    monkeypatch.setattr(G.GmailClient, "history_since", lambda self, h: iter([]))
    summary = sync_user(db_conn, settings, test_user)

    assert summary["new_messages"] == 0
    assert summary["total_messages"] == 3

    import pyarrow.parquet as pq
    rows = pq.read_table(messages_path(tmp_path, test_user)).to_pylist()
    assert len(rows) == 3                            # not 6


@pytest.mark.slow
def test_a_second_sync_actually_takes_the_incremental_path_not_just_dedup(
        db_conn, test_user, tmp_path, gmail_class_mocked, monkeypatch):
    # The bug this guards against: `G.sync` returns an updated cursor but
    # never persists it - if `sync_user` forgot to save it (it did, until
    # now), `state.history_id` loads empty every time and every sync,
    # incremental or not, re-fetches the entire mailbox via
    # `list_message_ids`/`raw_message`. The test above couldn't tell the
    # difference: dedup makes a full-refetch-that-finds-nothing-new look
    # identical to a real incremental sync from the outside. This one can -
    # a full refetch calls `list_message_ids`, a real incremental sync never
    # does.
    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    sync_user(db_conn, settings, test_user)          # first sync: saves the cursor

    def _fail_if_called(self, query="", max_messages=0, include_spam_trash=False):
        raise AssertionError(
            "list_message_ids was called - the sync fell back to a full "
            "re-fetch instead of using the saved historyId cursor")

    monkeypatch.setattr(G.GmailClient, "list_message_ids", _fail_if_called)
    monkeypatch.setattr(G.GmailClient, "history_since", lambda self, h: iter([]))

    summary = sync_user(db_conn, settings, test_user)   # must not raise above
    assert summary["new_messages"] == 0


@pytest.mark.slow
def test_backfill_history_writes_a_real_index_and_marks_itself_done(
        db_conn, test_user, tmp_path, gmail_class_mocked):
    from app.ingestion.worker import backfill_history

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status) VALUES (%s, 'ready')", (test_user,))

    summary = backfill_history(db_conn, settings, test_user)

    assert summary["new_messages"] == 3
    done = db_conn.execute(
        "SELECT full_history_synced FROM sync_state WHERE user_id = %s",
        (test_user,)).fetchone()[0]
    assert done is True


@pytest.mark.slow
def test_backfill_history_does_not_duplicate_what_a_prior_sync_already_has(
        db_conn, test_user, tmp_path, gmail_class_mocked):
    from app.ingestion.worker import backfill_history

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')", (test_user,))

    sync_user(db_conn, settings, test_user)              # the fast recent-only pass
    summary = backfill_history(db_conn, settings, test_user)   # sees the same 3 mocked ids

    assert summary["new_messages"] == 0                  # already had all of them
    assert summary["total_messages"] == 3                # not 6

    import pyarrow.parquet as pq
    rows = pq.read_table(messages_path(tmp_path, test_user)).to_pylist()
    assert len(rows) == 3


@pytest.mark.slow
def test_the_index_is_queryable_end_to_end(db_conn, test_user, tmp_path, gmail_class_mocked):
    from emailrag.pipeline import Pipeline

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    sync_user(db_conn, settings, test_user)

    pipe = Pipeline(index_dir(tmp_path, test_user), messages_path(tmp_path, test_user),
                    rerank="none", verbose=False)
    result = pipe.search("what did carol ask about the budget", top_n=3)

    assert any(m.sender == "carol@example.com" for m in result.messages)


@pytest.mark.slow
def test_a_second_larger_sync_does_not_leave_a_stale_chunk_text_cache(
        db_conn, test_user, tmp_path, gmail_class_mocked, monkeypatch):
    """The bug this guards against: `_rebuild_index` always rewrites
    `chunks.jsonl`/`dense`/`bm25` fresh off the current corpus, but used to
    leave `chunk_texts.jsonl.gz` (`emailrag.index.chunktext`'s cache,
    validated against `chunks.jsonl` on every `Pipeline` load) sitting on
    disk from whatever smaller corpus the *previous* sync built - the same
    shape of bug as the dense-vector shard checkpoint fixed above.
    Reproduced live: a real mailbox's second sync grew from 7,271 to 7,340
    chunks, and the very next /chat request raised `ParityError` ("7,340
    local chunk ids vs 7,271") instead of answering.
    """
    from emailrag.pipeline import Pipeline

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))

    sync_user(db_conn, settings, test_user)          # first sync: m1, m2, m3

    raw_4 = (b"From: dave@example.com\r\nTo: alice@example.com\r\n"
            b"Subject: New topic\r\nDate: Wed, 7 Jan 2026 09:00:00 -0000\r\n"
            b"Message-ID: <m4@example.com>\r\n\r\n"
            b"A brand new message that grows the corpus on the second sync.\r\n")
    raw_by_id = {**RAW_BY_ID, "m4": raw_4}
    monkeypatch.setattr(G.GmailClient, "history_since", lambda self, h: iter(["m4"]))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: raw_by_id[message_id])

    summary = sync_user(db_conn, settings, test_user)   # second sync: grows the corpus
    assert summary["new_messages"] == 1
    assert summary["total_messages"] == 4

    # This is the exact call that raised ParityError in production - it
    # must not, and it must see the new message.
    pipe = Pipeline(index_dir(tmp_path, test_user), messages_path(tmp_path, test_user),
                    rerank="none", verbose=False)
    result = pipe.search("brand new message that grows the corpus", top_n=3)
    assert any(m.sender == "dave@example.com" for m in result.messages)


@pytest.mark.slow
def test_a_sql_route_with_no_resolvable_window_still_falls_through_to_retrieval(
        db_conn, test_user, tmp_path, gmail_class_mocked):
    """Rules and the LLM classifier can both call a question "sql" with
    nothing for `parse_window` to resolve - "what are my deadlines" matches
    `_TEMPORAL`'s bare "deadline" via rules with no week/today/explicit-date
    phrase for `parse_window` to use, and "what's most urgent" reached the
    same state live through the LLM classifier ("implies sorting by due
    date"). Either way `_sql_arm`/`_message_date_arm` have nothing to filter
    on and return empty, and a pure-SQL route stopping there is a guaranteed
    empty answer for a question retrieval could actually have answered - see
    `pipeline.py`'s `search()`.
    """
    from emailrag.pipeline import Pipeline
    from emailrag.router.classify import RouterDecision

    settings = _settings(tmp_path)
    _seed_tokens(db_conn, test_user, settings.master_key)
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                    (test_user,))
    sync_user(db_conn, settings, test_user)

    pipe = Pipeline(index_dir(tmp_path, test_user), messages_path(tmp_path, test_user),
                    rerank="none", verbose=False, route=True, commitments=[])

    class _AlwaysSQL:
        """Stands in for either real decision path - what matters here is
        only that the route is "sql" with a question `parse_window` cannot
        resolve, not which mechanism produced it."""

        def route(self, question):
            return RouterDecision(query=question, route="sql",
                                  reason="forced for this test",
                                  confidence=0.9, decided_by="rules")

    pipe.router = _AlwaysSQL()

    result = pipe.search("what did carol ask about the budget", top_n=3)

    assert any(m.sender == "carol@example.com" for m in result.messages)


# -- keeping a returning user's mailbox from going stale -------------------
#
# Nothing else in this codebase ever enqueues `incremental_sync` on its own -
# only a brand-new login does, via `initial_sync`. Without this scheduler, a
# user's mailbox is frozen at whatever it looked like on their last login,
# forever, until they log out and back in.

def _utc_stamp(minutes_ago: int) -> str:
    # Matches exactly how the real sync path formats it
    # (corpus/gmail.py's `SyncState.last_sync_utc`) - computed in Python,
    # not via SQL date arithmetic against `now()`, which returns session-
    # local time (this test database's session is Asia/Kolkata) and would
    # need its own explicit `AT TIME ZONE 'UTC'` conversion to avoid writing
    # a wall-clock-local string with a falsely-claimed "Z" (UTC) suffix.
    from datetime import datetime, timedelta, timezone
    when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_schedule_due_syncs_enqueues_a_stale_ready_user(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, last_sync_utc) VALUES (%s, 'ready', %s)",
        (test_user, _utc_stamp(30)))

    scheduled = schedule_due_syncs(db_conn)

    assert str(test_user) in scheduled
    job = db_conn.execute(
        "SELECT type, status FROM jobs WHERE user_id = %s", (test_user,)).fetchone()
    assert job == ("incremental_sync", "queued")


def test_schedule_due_syncs_skips_a_recently_synced_user(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, last_sync_utc) VALUES (%s, 'ready', %s)",
        (test_user, _utc_stamp(2)))

    assert str(test_user) not in schedule_due_syncs(db_conn)


def test_schedule_due_syncs_skips_a_user_still_syncing(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, last_sync_utc) VALUES (%s, 'syncing', %s)",
        (test_user, _utc_stamp(30)))

    assert str(test_user) not in schedule_due_syncs(db_conn)


def test_schedule_due_syncs_does_not_duplicate_an_already_queued_job(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, last_sync_utc) VALUES (%s, 'ready', %s)",
        (test_user, _utc_stamp(30)))

    first = schedule_due_syncs(db_conn)
    second = schedule_due_syncs(db_conn)

    assert str(test_user) in first
    assert str(test_user) not in second      # already queued, not doubled up
    count = db_conn.execute(
        "SELECT count(*) FROM jobs WHERE user_id = %s AND type = 'incremental_sync'",
        (test_user,)).fetchone()[0]
    assert count == 1
