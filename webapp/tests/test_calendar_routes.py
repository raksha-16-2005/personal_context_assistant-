from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.calendar import client as calendar_client_module
from app.commitments import generate_calendar_suggestions, insert_commitments
from app.config import Settings
from app.deps import get_current_user_id, get_settings
from app.main import app
from app.tokens.store import DbTokenStore
from emailrag.extraction.schema import Commitment

DATABASE_URL = "postgresql:///emailrag"


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
def client(tmp_path, test_user):
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_current_user_id] = lambda: test_user
    with TestClient(app, follow_redirects=False) as c:
        yield c, test_user, settings
    app.dependency_overrides.clear()


def _seed_suggestion(db_conn, user_id: str, **overrides) -> str:
    base = dict(message_id="m1", text="send the agenda", kind="deliverable",
               direction="i_owe", owner="alice@example.com",
               counterparty="bob@example.com", due_phrase="Friday",
               due_at=date(2026, 8, 14), model="gemini-2.5-flash")
    base.update(overrides)
    insert_commitments(db_conn, user_id, [Commitment(**base)])
    generate_calendar_suggestions(db_conn, user_id)
    return db_conn.execute(
        "SELECT id FROM calendar_suggestions WHERE user_id = %s", (user_id,)
    ).fetchone()[0]


def test_list_suggestions_defaults_to_pending(client, db_conn):
    c, user_id, _ = client
    suggestion_id = _seed_suggestion(db_conn, user_id)

    resp = c.get("/calendar/suggestions")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["id"] == str(suggestion_id)
    assert body[0]["status"] == "pending"
    assert body[0]["text"] == "send the agenda"
    assert body[0]["due_at"] == "2026-08-14"
    # The commitment's source message - lets the UI offer "view full email"
    # (GET /messages/{id}) the same way a chat citation does.
    assert body[0]["message_id"] == "m1"


def test_list_suggestions_excludes_dismissed_by_default(client, db_conn):
    c, user_id, _ = client
    suggestion_id = _seed_suggestion(db_conn, user_id)
    db_conn.execute(
        "UPDATE calendar_suggestions SET status = 'dismissed' WHERE id = %s",
        (suggestion_id,))

    resp = c.get("/calendar/suggestions")
    assert resp.json() == []

    resp_all = c.get("/calendar/suggestions", params={"status": "all"})
    assert len(resp_all.json()) == 1


def test_confirm_creates_a_real_event_and_updates_status(client, db_conn, monkeypatch):
    c, user_id, settings = client
    suggestion_id = _seed_suggestion(db_conn, user_id)
    DbTokenStore(conn=db_conn, user_id=user_id, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()

    calls = []

    def fake_create_event(self, summary, description, due_date):
        calls.append((summary, description, due_date))
        return {"id": "google-event-123"}

    monkeypatch.setattr(calendar_client_module.CalendarClient, "create_event",
                        fake_create_event)

    resp = c.post(f"/calendar/suggestions/{suggestion_id}/confirm")

    assert resp.status_code == 200
    assert resp.json() == {"status": "confirmed", "calendar_event_id": "google-event-123"}
    assert calls == [("send the agenda",
                      "alice@example.com -> bob@example.com: send the agenda",
                      date(2026, 8, 14))]

    row = db_conn.execute(
        "SELECT status, calendar_event_id FROM calendar_suggestions WHERE id = %s",
        (suggestion_id,)).fetchone()
    assert row == ("confirmed", "google-event-123")


def test_confirm_is_rejected_once_already_confirmed(client, db_conn, monkeypatch):
    c, user_id, settings = client
    suggestion_id = _seed_suggestion(db_conn, user_id)
    DbTokenStore(conn=db_conn, user_id=user_id, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    monkeypatch.setattr(calendar_client_module.CalendarClient, "create_event",
                        lambda self, summary, description, due_date: {"id": "e1"})

    first = c.post(f"/calendar/suggestions/{suggestion_id}/confirm")
    assert first.status_code == 200

    second = c.post(f"/calendar/suggestions/{suggestion_id}/confirm")
    assert second.status_code == 409


def test_dismiss_marks_it_dismissed_without_calling_google(client, db_conn, monkeypatch):
    c, user_id, _ = client
    suggestion_id = _seed_suggestion(db_conn, user_id)

    def fail_if_called(*a, **k):
        raise AssertionError("dismiss must never call the Calendar API")
    monkeypatch.setattr(calendar_client_module.CalendarClient, "create_event", fail_if_called)

    resp = c.post(f"/calendar/suggestions/{suggestion_id}/dismiss")

    assert resp.status_code == 200
    assert resp.json() == {"status": "dismissed"}
    status = db_conn.execute(
        "SELECT status FROM calendar_suggestions WHERE id = %s", (suggestion_id,)
    ).fetchone()[0]
    assert status == "dismissed"


def test_confirming_someone_elses_suggestion_is_not_found(client, db_conn):
    c, user_id, _ = client
    other = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        ("calendar-test-other-sub", "other@example.com")).fetchone()[0]
    try:
        other_suggestion_id = _seed_suggestion(db_conn, str(other))
        resp = c.post(f"/calendar/suggestions/{other_suggestion_id}/confirm")
        assert resp.status_code == 404
    finally:
        db_conn.execute("DELETE FROM users WHERE id = %s", (other,))
