from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import Settings
from app.ingestion.worker import sync_user
from app.pipeline_pool import PipelinePool
from app.tokens.store import DbTokenStore
from emailrag.corpus import gmail as G

RAW_A = (b"From: dave@example.com\r\nTo: erin@example.com\r\n"
        b"Subject: Roadmap\r\nDate: Mon, 5 Jan 2026 09:00:00 -0000\r\n"
        b"Message-ID: <a1@example.com>\r\n\r\n"
        b"Here is the roadmap draft for review.\r\n")
RAW_B = (b"From: frank@example.com\r\nTo: grace@example.com\r\n"
        b"Subject: Invoice\r\nDate: Tue, 6 Jan 2026 09:00:00 -0000\r\n"
        b"Message-ID: <b1@example.com>\r\n\r\n"
        b"Please find attached invoice for last month.\r\n")


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


def _seed_and_sync(db_conn, settings, user_id, raw_message, monkeypatch):
    DbTokenStore(conn=db_conn, user_id=user_id, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')", (user_id,))

    monkeypatch.setattr(
        G.GmailClient, "list_message_ids",
        lambda self, query="", max_messages=0, include_spam_trash=False: iter(["m1"]))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: raw_message)
    monkeypatch.setattr(G.GmailClient, "current_history_id", lambda self: "1000")
    sync_user(db_conn, settings, user_id)


@pytest.fixture()
def second_user(db_conn):
    row = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        (f"test-sub-second-{id(object())}", "second@example.com"),
    ).fetchone()
    user_id = str(row[0])
    yield user_id
    db_conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


@pytest.mark.slow
def test_two_users_pipelines_share_one_model_instance(
        db_conn, test_user, second_user, tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    _seed_and_sync(db_conn, settings, test_user, RAW_A, monkeypatch)
    _seed_and_sync(db_conn, settings, second_user, RAW_B, monkeypatch)

    pool = PipelinePool(tmp_path, settings.shipped_chunking, settings.shipped_model,
                        settings.shipped_rerank, max_cached=8)
    pipe_a = pool.get(test_user)
    pipe_b = pool.get(second_user)

    # The whole point of PipelinePool: two users, one loaded model in memory.
    assert pipe_a.model is pipe_b.model
    assert pipe_a is not pipe_b
    # Same reasoning applies to the reranker - it was the second model each
    # cold Pipeline load used to pay for on its own before this was shared.
    assert pipe_a.reranker is pipe_b.reranker
    assert pipe_a.reranker is not None


RAW_INVOICE_TODAY = (
    b"From: frank@example.com\r\nTo: grace@example.com\r\n"
    b"Subject: Invoice\r\nDate: Tue, 6 Jan 2026 09:00:00 -0000\r\n"
    b"Message-ID: <b1@example.com>\r\n\r\n"
    b"Please find attached invoice for last month.\r\n")
RAW_NEWSLETTER_TODAY = (
    b"From: updates@newsletter.example.com\r\nTo: grace@example.com\r\n"
    b"Subject: Weekly newsletter\r\nDate: Tue, 6 Jan 2026 11:00:00 -0000\r\n"
    b"Message-ID: <b2@example.com>\r\n"
    b"List-Unsubscribe: <mailto:unsub@newsletter.example.com>\r\n\r\n"
    b"Check out this week's top stories.\r\n")


@pytest.mark.slow
def test_the_date_arm_reports_the_honest_total_including_filtered_bulk_mail(
        db_conn, test_user, tmp_path, monkeypatch):
    """`_message_date_arm` (pipeline.py) undercounts "how many" the moment a
    caller never tells it about the mail `corpus.filters.is_bulk` already
    dropped at ingestion - this is the fix, exercised through the same
    `PipelinePool` the web app actually uses so `bulk_sample` really is wired
    in, not just passed in isolation.
    """
    settings = _settings(tmp_path)
    by_id = {"b1": RAW_INVOICE_TODAY, "b2": RAW_NEWSLETTER_TODAY}
    DbTokenStore(conn=db_conn, user_id=test_user, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')", (test_user,))
    monkeypatch.setattr(
        G.GmailClient, "list_message_ids",
        lambda self, query="", max_messages=0, include_spam_trash=False: iter(by_id.keys()))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: by_id[message_id])
    monkeypatch.setattr(G.GmailClient, "current_history_id", lambda self: "1000")
    sync_user(db_conn, settings, test_user)

    pool = PipelinePool(tmp_path, settings.shipped_chunking, settings.shipped_model,
                        settings.shipped_rerank, max_cached=8)
    pipe = pool.get(test_user)
    assert pipe.bulk_sample is not None

    result = pipe.search("how many messages did I get today", as_of="2026-01-06")

    note = result.route["date_note"]
    assert "1 non-bulk message" in note
    assert "plus 1 promotional/newsletter message" in note
    assert "(2 total received)" in note


class _FakePipeline:
    instances_built = 0

    def __init__(self, *args, **kwargs):
        _FakePipeline.instances_built += 1
        self.build_number = _FakePipeline.instances_built


@pytest.fixture()
def fake_pipeline(monkeypatch):
    # Isolates the pool's own bookkeeping (cache hit/miss, LRU eviction,
    # invalidation) from Pipeline's real cost - the one test above already
    # proves the real model-sharing behaviour end to end.
    _FakePipeline.instances_built = 0
    monkeypatch.setattr("app.pipeline_pool.Pipeline", _FakePipeline)
    monkeypatch.setattr("emailrag.index.embed.load_model", lambda model_id: "shared-model-object")
    return _FakePipeline


def test_get_returns_the_same_cached_instance(tmp_path, fake_pipeline):
    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=8)
    first = pool.get("user-1")
    second = pool.get("user-1")

    assert first is second
    assert fake_pipeline.instances_built == 1


def test_lru_eviction_drops_the_least_recently_used(tmp_path, fake_pipeline):
    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=1)
    first = pool.get("user-1")
    pool.get("user-2")                          # over capacity -> evicts user-1
    assert len(pool) == 1

    again = pool.get("user-1")                  # cache miss -> rebuilt
    assert again is not first
    assert fake_pipeline.instances_built == 3


def test_getting_a_cached_user_again_does_not_count_as_the_lru_victim(tmp_path, fake_pipeline):
    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=2)
    pool.get("user-1")
    pool.get("user-2")
    pool.get("user-1")                          # touches user-1 -> now most-recent
    pool.get("user-3")                          # over capacity -> evicts user-2, not user-1

    assert list(pool._cache.keys()) == ["user-1", "user-3"]


def test_invalidate_forces_a_reload(tmp_path, fake_pipeline):
    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=8)
    first = pool.get("user-1")
    pool.invalidate("user-1")
    second = pool.get("user-1")

    assert first is not second


def test_the_shared_model_is_loaded_only_once(tmp_path, fake_pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr("emailrag.index.embed.load_model",
                        lambda model_id: calls.append(model_id) or "shared-model-object")

    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=8)
    pool.get("user-1")
    pool.get("user-2")
    pool.get("user-3")

    assert calls == ["fake-model"]              # not once per user


def test_warm_loads_the_model_before_any_get_call(tmp_path, fake_pipeline, monkeypatch):
    calls = []
    monkeypatch.setattr("emailrag.index.embed.load_model",
                        lambda model_id: calls.append(model_id) or "shared-model-object")

    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=8)
    assert calls == []                          # nothing loaded yet

    pool.warm()

    assert calls == ["fake-model"]              # loaded before any user ever connects
    pool.get("user-1")
    assert calls == ["fake-model"]              # get() reuses it, doesn't load again


def test_warm_is_a_noop_for_an_unconfigured_rerank_spec(tmp_path, fake_pipeline):
    pool = PipelinePool(tmp_path, "thread_aware", "fake-model", "none", max_cached=8)
    pool.warm()                                 # "none" has no CrossEncoderReranker spec
    assert pool._reranker is None
