from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import Settings
from app.deps import get_current_user_id, get_pipeline_pool, get_settings
from app.gemini_keys import save_gemini_key
from app.ingestion.worker import sync_user
from app.main import app
from app.pipeline_pool import PipelinePool
from app.tokens.store import DbTokenStore
from emailrag.corpus import gmail as G

DATABASE_URL = "postgresql:///emailrag"

RAW = (b"From: alice@example.com\r\nTo: bob@example.com\r\n"
      b"Subject: Project kickoff\r\nDate: Mon, 5 Jan 2026 09:00:00 -0000\r\n"
      b"Message-ID: <m1@example.com>\r\n\r\n"
      b"Let's kick off the project next week. I'll send the agenda by Friday.\r\n")


def _settings(tmp_path) -> Settings:
    return Settings(
        database_url=DATABASE_URL,
        master_key=Fernet.generate_key().decode(),
        session_secret="test-session-secret",
        gmail_client_id="test-client-id",
        gmail_client_secret="test-client-secret",
        oauth_redirect_base_url="http://localhost:8000",
        user_index_root=tmp_path,
        shipped_chunking="thread_aware",
        shipped_model="sentence-transformers/all-MiniLM-L6-v2",
        shipped_rerank="none",
    )


@pytest.fixture()
def synced_user(db_conn, test_user, tmp_path, monkeypatch):
    """A user with a real per-user index already built, ready for /chat."""
    settings = _settings(tmp_path)
    DbTokenStore(conn=db_conn, user_id=test_user, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')", (test_user,))

    monkeypatch.setattr(
        G.GmailClient, "list_message_ids",
        lambda self, query="", max_messages=0, include_spam_trash=False: iter(["m1"]))
    monkeypatch.setattr(G.GmailClient, "raw_message", lambda self, message_id: RAW)
    monkeypatch.setattr(G.GmailClient, "current_history_id", lambda self: "1000")
    sync_user(db_conn, settings, test_user)

    return test_user, settings


@pytest.fixture()
def client(synced_user):
    user_id, settings = synced_user
    pool = PipelinePool(settings.user_index_root, settings.shipped_chunking,
                        settings.shipped_model, settings.shipped_rerank)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_pipeline_pool] = lambda: pool
    with TestClient(app, follow_redirects=False) as c:
        yield c, user_id, settings
    app.dependency_overrides.clear()


def _mock_gemini_response(monkeypatch, text: str):
    import json as _json

    from emailrag.llm import client as C

    def fake_urlopen(req, timeout=None):
        body = _json.dumps(
            {"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode()
        return io.BytesIO(body)

    class _Ctx:
        def __init__(self, obj):
            self.obj = obj

        def __enter__(self):
            return self.obj

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(C.urllib.request, "urlopen",
                        lambda req, timeout=None: _Ctx(fake_urlopen(req, timeout)))


@pytest.mark.slow
def test_chat_without_a_gemini_key_is_rejected(client):
    c, _, _ = client
    resp = c.post("/chat", json={"question": "what is the kickoff about"})
    assert resp.status_code == 400
    assert "Gemini API key" in resp.json()["detail"]


@pytest.mark.slow
def test_chat_before_sync_is_ready_is_rejected(client, db_conn):
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    db_conn.execute("UPDATE sync_state SET status = 'syncing' WHERE user_id = %s", (user_id,))

    resp = c.post("/chat", json={"question": "what is the kickoff about"})
    assert resp.status_code == 409


@pytest.mark.slow
def test_chat_answers_with_a_real_citation_and_persists_the_turn(
        client, db_conn, monkeypatch):
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_response(
        monkeypatch, "The kickoff is next week and the agenda arrives Friday [1].")

    resp = c.post("/chat", json={"question": "when is the project kickoff"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is False
    assert "[1]" in data["answer"]
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["sender"] == "alice@example.com"

    conversation_id = data["conversation_id"]
    history = c.get(f"/conversations/{conversation_id}")
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()]
    assert roles == ["user", "assistant"]

    listing = c.get("/conversations")
    assert any(row["id"] == conversation_id for row in listing.json())


@pytest.mark.slow
def test_a_refusal_is_persisted_and_returned_in_history(client, db_conn, monkeypatch):
    """The frontend renders a natural sentence instead of the literal
    INSUFFICIENT_CONTEXT sentinel for a refused answer (see Chat.jsx) - it
    needs `refused` from history, not just from the live POST /chat
    response, or a reloaded conversation would still show the raw sentinel.
    """
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_response(monkeypatch, "INSUFFICIENT_CONTEXT")

    resp = c.post("/chat", json={"question": "what is my flight itinerary"})
    assert resp.json()["refused"] is True

    conversation_id = resp.json()["conversation_id"]
    history = c.get(f"/conversations/{conversation_id}").json()
    assistant_turn = next(m for m in history if m["role"] == "assistant")
    assert assistant_turn["refused"] is True

    user_turn = next(m for m in history if m["role"] == "user")
    assert user_turn["refused"] is False


@pytest.mark.slow
def test_a_second_message_reuses_the_same_conversation(client, db_conn, monkeypatch):
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_response(monkeypatch, "It's next week [1].")

    first = c.post("/chat", json={"question": "when is the kickoff"})
    conversation_id = first.json()["conversation_id"]

    second = c.post("/chat", json={
        "question": "who is sending the agenda", "conversation_id": conversation_id})

    assert second.json()["conversation_id"] == conversation_id
    history = c.get(f"/conversations/{conversation_id}").json()
    assert len(history) == 4          # 2 user turns + 2 assistant turns


@pytest.mark.slow
def test_get_message_returns_the_full_body_behind_a_citation(
        client, db_conn, monkeypatch):
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_response(monkeypatch, "The kickoff is next week [1].")

    chat_resp = c.post("/chat", json={"question": "when is the kickoff"})
    message_id = chat_resp.json()["citations"][0]["message_id"]

    resp = c.get(f"/messages/{message_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sender"] == "alice@example.com"
    assert data["subject"] == "Project kickoff"
    assert "agenda by Friday" in data["body"]


@pytest.mark.slow
def test_get_message_404s_for_an_unknown_id(client):
    c, _, _ = client
    resp = c.get("/messages/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.slow
def test_a_conversation_id_from_another_user_is_rejected(client, db_conn):
    c, user_id, settings = client
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")

    other = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        ("someone-elses-sub", "other@example.com")).fetchone()[0]
    other_conv = db_conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (%s, 'x') RETURNING id",
        (other,)).fetchone()[0]

    try:
        resp = c.post("/chat", json={
            "question": "anything", "conversation_id": str(other_conv)})
        assert resp.status_code == 404
    finally:
        db_conn.execute("DELETE FROM users WHERE id = %s", (other,))


def test_deleting_a_conversation_removes_it_and_its_messages(client, db_conn):
    c, user_id, _ = client
    conv_id = db_conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (%s, 'x') RETURNING id",
        (user_id,)).fetchone()[0]
    db_conn.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', 'hi')",
        (conv_id,))

    resp = c.delete(f"/conversations/{conv_id}")
    assert resp.status_code == 204

    assert db_conn.execute(
        "SELECT 1 FROM conversations WHERE id = %s", (conv_id,)).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM messages WHERE conversation_id = %s", (conv_id,)).fetchone() is None
    assert c.get("/conversations").json() == []


def test_deleting_another_users_conversation_404s(client, db_conn):
    c, _, _ = client
    other = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        ("someone-elses-sub-2", "other2@example.com")).fetchone()[0]
    other_conv = db_conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (%s, 'x') RETURNING id",
        (other,)).fetchone()[0]

    try:
        resp = c.delete(f"/conversations/{other_conv}")
        assert resp.status_code == 404
        assert db_conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s", (other_conv,)).fetchone() is not None
    finally:
        db_conn.execute("DELETE FROM users WHERE id = %s", (other,))
