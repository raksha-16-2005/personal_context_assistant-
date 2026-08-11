from __future__ import annotations

import io
import json as _json
import sys
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.config import Settings
from app.extraction.worker import extract_for_user
from app.gemini_keys import save_gemini_key
from app.ingestion.worker import sync_user
from app.tokens.store import DbTokenStore
from emailrag.corpus import gmail as G

# Extraction only scans a recent window (see EXTRACTION_WINDOW_DAYS) - the
# send date has to be "now", not a fixed date that ages out of that window
# as real time passes.
_SENT = format_datetime(datetime.now(timezone.utc)).encode()
RAW = (b"From: alice@example.com\r\nTo: bob@example.com\r\n"
      b"Subject: Project kickoff\r\nDate: " + _SENT + b"\r\n"
      b"Message-ID: <m1@example.com>\r\n\r\n"
      b"Let's kick off the project. I'll send the agenda by Friday.\r\n")


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
        shipped_rerank="none",
    )


@pytest.fixture()
def synced_user(db_conn, test_user, tmp_path, monkeypatch):
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


def _mock_gemini_json(monkeypatch, payload: dict):
    from emailrag.llm import client as C

    text = _json.dumps(payload)

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
def test_extraction_is_a_noop_without_a_gemini_key(synced_user, db_conn):
    user_id, settings = synced_user
    result = extract_for_user(db_conn, settings, user_id)
    assert result == {"skipped": "no gemini key saved"}


@pytest.mark.slow
def test_extraction_is_a_noop_before_any_sync(db_conn, test_user, tmp_path):
    settings = _settings(tmp_path)
    save_gemini_key(db_conn, test_user, settings.master_key, "fake-key")
    result = extract_for_user(db_conn, settings, test_user)
    assert result == {"skipped": "no messages synced yet"}


@pytest.mark.slow
def test_extraction_stores_a_commitment_and_a_calendar_suggestion(
        synced_user, db_conn, monkeypatch):
    user_id, settings = synced_user
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_json(monkeypatch, {"commitments": [
        {"text": "send the agenda", "kind": "deliverable", "direction": "i_owe",
         "owner": "alice@example.com", "counterparty": "bob@example.com",
         "due_phrase": "Friday", "confidence": 0.9, "quote": "I'll send the agenda by Friday"},
    ]})

    result = extract_for_user(db_conn, settings, user_id)

    assert result["messages_scanned"] == 1
    assert result["commitments_found"] == 1
    assert result["commitments_new"] == 1
    assert result["suggestions_created"] == 1

    commitment = db_conn.execute(
        "SELECT text, owner, due_at FROM commitments WHERE user_id = %s", (user_id,)
    ).fetchone()
    assert commitment[0] == "send the agenda"
    assert commitment[1] == "alice@example.com"
    assert commitment[2] is not None       # "Friday" resolved to a real date

    suggestion = db_conn.execute(
        "SELECT status FROM calendar_suggestions WHERE user_id = %s", (user_id,)
    ).fetchone()
    assert suggestion == ("pending",)


@pytest.mark.slow
def test_re_extracting_the_same_window_does_not_duplicate(synced_user, db_conn, monkeypatch):
    user_id, settings = synced_user
    save_gemini_key(db_conn, user_id, settings.master_key, "fake-gemini-key")
    _mock_gemini_json(monkeypatch, {"commitments": [
        {"text": "send the agenda", "kind": "deliverable", "direction": "i_owe",
         "owner": "alice@example.com", "due_phrase": "Friday", "confidence": 0.9},
    ]})

    extract_for_user(db_conn, settings, user_id)
    second = extract_for_user(db_conn, settings, user_id)

    assert second["commitments_new"] == 0
    count = db_conn.execute(
        "SELECT count(*) FROM commitments WHERE user_id = %s", (user_id,)
    ).fetchone()[0]
    assert count == 1
