from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth.session import create_session_cookie
from app.config import Settings
from app.deps import get_current_user_id, get_pipeline_pool, get_settings
from app.gemini_keys import save_gemini_key
from app.main import app
from app.pipeline_pool import PipelinePool
from app.tokens.store import DbTokenStore

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
    pool = PipelinePool(settings.user_index_root, settings.shipped_chunking,
                        settings.shipped_model, settings.shipped_rerank)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_current_user_id] = lambda: test_user
    app.dependency_overrides[get_pipeline_pool] = lambda: pool
    with TestClient(app, follow_redirects=False) as c:
        yield c, test_user, settings, pool
    app.dependency_overrides.clear()


def test_me_returns_the_logged_in_users_id_and_email(client, db_conn):
    c, user_id, _, _ = client
    email = db_conn.execute(
        "SELECT email FROM users WHERE id = %s", (user_id,)).fetchone()[0]

    resp = c.get("/me")

    assert resp.status_code == 200
    assert resp.json() == {"id": user_id, "email": email, "timezone": "UTC"}


def test_updating_the_timezone_is_reflected_in_me(client):
    c, _, _, _ = client
    resp = c.put("/account/timezone", json={"timezone": "Asia/Kolkata"})
    assert resp.status_code == 204

    assert c.get("/me").json()["timezone"] == "Asia/Kolkata"


def test_sync_status_with_no_row_yet_reads_as_pending(client):
    c, _, _, _ = client
    resp = c.get("/sync-status")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "pending", "messages_seen": 0, "full_history_synced": False,
        "error_detail": "", "progress_current": 0, "progress_total": 0,
        "eta_seconds": 60, "eta_is_estimate": True,
    }


def test_sync_status_reflects_sync_state_and_zeroes_eta_once_ready(client, db_conn):
    c, user_id, _, _ = client
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, messages_seen) VALUES (%s, 'syncing', 3)",
        (user_id,))
    status = c.get("/sync-status").json()
    assert status["eta_seconds"] == 60          # no progress yet - the flat fallback
    assert status["eta_is_estimate"] is True

    db_conn.execute(
        "UPDATE sync_state SET status = 'ready', messages_seen = 9 WHERE user_id = %s",
        (user_id,))
    resp = c.get("/sync-status").json()
    assert resp["status"] == "ready"
    assert resp["messages_seen"] == 9
    assert resp["eta_seconds"] == 0
    assert resp["eta_is_estimate"] is False


def test_sync_status_projects_a_real_eta_from_live_progress(client, db_conn):
    c, user_id, _, _ = client
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, sync_started_at, "
        "progress_current, progress_total) "
        "VALUES (%s, 'syncing', now() - interval '10 seconds', 5, 20)",
        (user_id,))

    resp = c.get("/sync-status").json()
    assert resp["progress_current"] == 5
    assert resp["progress_total"] == 20
    assert resp["eta_is_estimate"] is False
    # ~10s for 5 messages -> ~2s/message -> ~30s for the remaining 15,
    # generous bounds for test timing jitter.
    assert 20 <= resp["eta_seconds"] <= 45


def test_sync_status_surfaces_error_detail(client, db_conn):
    c, user_id, _, _ = client
    db_conn.execute(
        "INSERT INTO sync_state (user_id, status, error_detail) "
        "VALUES (%s, 'error', 'refresh token revoked')", (user_id,))
    assert c.get("/sync-status").json()["error_detail"] == "refresh token revoked"


def test_an_unresolvable_timezone_is_rejected(client, db_conn):
    c, user_id, _, _ = client
    resp = c.put("/account/timezone", json={"timezone": "Mars/Olympus_Mons"})

    assert resp.status_code == 400
    # Rejected before it ever reaches the row - still UTC, not silently set
    # to garbage that would then break every "today" for this user.
    tz = db_conn.execute(
        "SELECT timezone FROM users WHERE id = %s", (user_id,)).fetchone()[0]
    assert tz == "UTC"


def test_gemini_key_status_starts_false(client):
    c, _, _, _ = client
    resp = c.get("/account/gemini-key")
    assert resp.json() == {"has_key": False, "has_key_2": False}


def test_setting_the_gemini_key_flips_status_and_never_echoes_the_key(client, db_conn):
    c, user_id, settings, _ = client
    resp = c.put("/account/gemini-key", json={"api_key": "a-secret-key"})
    assert resp.status_code == 204
    assert "a-secret-key" not in resp.text

    status = c.get("/account/gemini-key")
    assert status.json() == {"has_key": True, "has_key_2": False}

    row = db_conn.execute(
        "SELECT encrypted_key FROM gemini_keys WHERE user_id = %s", (user_id,)).fetchone()
    assert b"a-secret-key" not in bytes(row[0])


def test_a_second_gemini_key_is_optional_and_flips_its_own_status(client, db_conn):
    c, user_id, _, _ = client
    c.put("/account/gemini-key", json={"api_key": "primary-key"})
    assert c.get("/account/gemini-key").json() == {"has_key": True, "has_key_2": False}

    resp = c.put("/account/gemini-key",
                 json={"api_key": "primary-key", "api_key_2": "backup-key"})
    assert resp.status_code == 204
    assert "backup-key" not in resp.text
    assert c.get("/account/gemini-key").json() == {"has_key": True, "has_key_2": True}

    row = db_conn.execute(
        "SELECT encrypted_key_2 FROM gemini_keys WHERE user_id = %s", (user_id,)).fetchone()
    assert b"backup-key" not in bytes(row[0])


def test_saving_without_a_second_key_leaves_an_existing_backup_key_alone(client):
    c, _, _, _ = client
    c.put("/account/gemini-key", json={"api_key": "primary-key", "api_key_2": "backup-key"})
    assert c.get("/account/gemini-key").json() == {"has_key": True, "has_key_2": True}

    # Re-saving just the primary (the common "I rotated my key" case) must
    # not silently wipe the backup - there's no way to resend a key that
    # was never echoed back in the first place.
    resp = c.put("/account/gemini-key", json={"api_key": "new-primary-key"})
    assert resp.status_code == 204
    assert c.get("/account/gemini-key").json() == {"has_key": True, "has_key_2": True}


def test_deleting_the_second_key_leaves_the_primary_alone(client):
    c, _, _, _ = client
    c.put("/account/gemini-key", json={"api_key": "primary-key", "api_key_2": "backup-key"})

    resp = c.delete("/account/gemini-key-2")
    assert resp.status_code == 204
    assert c.get("/account/gemini-key").json() == {"has_key": True, "has_key_2": False}


def test_setting_the_gemini_key_enqueues_extraction(client, db_conn):
    c, user_id, _, _ = client
    db_conn.execute("DELETE FROM jobs WHERE user_id = %s", (user_id,))

    resp = c.put("/account/gemini-key", json={"api_key": "a-secret-key"})

    assert resp.status_code == 204
    job = db_conn.execute(
        "SELECT type, status FROM jobs WHERE user_id = %s", (user_id,)).fetchone()
    assert job == ("extract_commitments", "queued")


def test_deleting_the_gemini_key_flips_status_back(client):
    c, _, _, _ = client
    c.put("/account/gemini-key", json={"api_key": "a-secret-key"})
    resp = c.delete("/account/gemini-key")
    assert resp.status_code == 204
    assert c.get("/account/gemini-key").json() == {"has_key": False, "has_key_2": False}


def test_delete_account_removes_the_user_row_and_cascades(client, db_conn):
    c, user_id, settings, _ = client
    DbTokenStore(conn=db_conn, user_id=user_id, master_key=settings.master_key, data={
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    save_gemini_key(db_conn, user_id, settings.master_key, "a-key")
    db_conn.execute(
        "INSERT INTO conversations (user_id, title) VALUES (%s, 'x')", (user_id,))

    resp = c.delete("/account")

    assert resp.status_code == 204
    assert db_conn.execute(
        "SELECT 1 FROM users WHERE id = %s", (user_id,)).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM oauth_tokens WHERE user_id = %s", (user_id,)).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM gemini_keys WHERE user_id = %s", (user_id,)).fetchone() is None
    assert db_conn.execute(
        "SELECT 1 FROM conversations WHERE user_id = %s", (user_id,)).fetchone() is None


def test_delete_account_removes_the_index_directory(client):
    c, user_id, settings, _ = client
    user_dir = settings.user_index_root / user_id
    (user_dir / "index").mkdir(parents=True)
    (user_dir / "messages.parquet").write_bytes(b"not real parquet, just a marker")
    assert user_dir.exists()

    resp = c.delete("/account")

    assert resp.status_code == 204
    assert not user_dir.exists()


def test_delete_account_is_fine_with_no_index_directory_on_disk(client):
    c, _, _, _ = client
    resp = c.delete("/account")           # never synced - directory never existed
    assert resp.status_code == 204


def test_delete_account_drops_the_cached_pipeline(client):
    c, user_id, _, pool = client
    pool._cache[user_id] = object()       # stand in for a real cached Pipeline
    c.delete("/account")
    assert user_id not in pool._cache


def test_delete_account_clears_the_session_cookie(client):
    c, _, _, _ = client
    resp = c.delete("/account")
    set_cookie = resp.headers.get("set-cookie", "")
    assert "emailrag_session=" in set_cookie
    assert "Max-Age=0" in set_cookie      # expired immediately, not the live session value


def test_delete_account_without_a_session_is_unauthorized(tmp_path):
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app, follow_redirects=False) as c:
            resp = c.delete("/account")
            assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_a_real_session_cookie_authorizes_the_delete(tmp_path, db_conn, test_user):
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app, follow_redirects=False) as c:
            cookie = create_session_cookie(settings.session_secret, test_user)
            c.cookies.set("emailrag_session", cookie)
            resp = c.delete("/account")
            assert resp.status_code == 204
            assert db_conn.execute(
                "SELECT 1 FROM users WHERE id = %s", (test_user,)).fetchone() is None
    finally:
        app.dependency_overrides.clear()
