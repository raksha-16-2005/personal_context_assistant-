from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth import routes as routes_module
from app.auth.session import create_state_token
from app.config import Settings
from app.db import connect
from app.deps import get_settings
from app.main import app

DATABASE_URL = "postgresql:///emailrag"


def _settings(**overrides) -> Settings:
    base = dict(
        database_url=DATABASE_URL,
        master_key=Fernet.generate_key().decode(),
        session_secret="test-session-secret",
        gmail_client_id="test-client-id",
        gmail_client_secret="test-client-secret",
        oauth_redirect_base_url="http://localhost:8000",
        user_index_root=Path("/tmp/emailrag-users"),
        shipped_chunking="thread_aware",
        shipped_model="sentence-transformers/all-MiniLM-L6-v2",
        shipped_rerank="L2@20/t192",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def client():
    settings = _settings()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app, base_url="http://testserver", follow_redirects=False) as c:
        yield c, settings
    app.dependency_overrides.clear()


def _cleanup_user(google_sub: str):
    with connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM users WHERE google_sub = %s", (google_sub,))


def test_login_redirects_to_google_with_the_right_scopes_and_a_state(client):
    c, _ = client
    resp = c.get("/auth/google/login")

    assert resp.status_code in (302, 307)
    location = resp.headers["location"]
    assert "accounts.google.com" in location
    assert "gmail.readonly" in location
    assert "calendar.events" in location
    assert "openid" in location
    assert "state=" in location


def test_callback_with_no_code_is_rejected(client):
    c, _ = client
    resp = c.get("/auth/google/callback")
    assert resp.status_code == 400


def test_callback_rejects_a_forged_state(client):
    c, _ = client
    resp = c.get("/auth/google/callback", params={"code": "abc", "state": "forged"})
    assert resp.status_code == 400
    assert "state" in resp.json()["detail"]


def test_callback_surfaces_googles_error_param(client):
    c, _ = client
    resp = c.get("/auth/google/callback", params={"error": "access_denied"})
    assert resp.status_code == 400
    assert "access_denied" in resp.json()["detail"]


def test_callback_without_a_refresh_token_names_the_fix(client, monkeypatch):
    c, settings = client
    state = create_state_token(settings.session_secret, "nonce")
    monkeypatch.setattr(routes_module.G, "exchange_code",
                        lambda *a, **k: {"access_token": "a"})   # no refresh_token

    resp = c.get("/auth/google/callback", params={"code": "abc", "state": state})

    assert resp.status_code == 400
    assert "myaccount.google.com/permissions" in resp.json()["detail"]


def test_callback_creates_a_user_stores_tokens_and_queues_the_first_sync(client, monkeypatch):
    c, settings = client
    google_sub = "sub-first-login-test"
    try:
        state = create_state_token(settings.session_secret, "nonce")
        monkeypatch.setattr(routes_module.G, "exchange_code", lambda *a, **k: {
            "refresh_token": "r-secret", "access_token": "a-secret",
            "expires_in": 3600, "scope": "gmail.readonly calendar.events openid email",
        })
        monkeypatch.setattr(routes_module, "fetch_userinfo", lambda access_token: {
            "sub": google_sub, "email": "new-user@example.com"})

        resp = c.get("/auth/google/callback", params={"code": "abc", "state": state})

        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/"
        set_cookie = resp.headers.get("set-cookie", "")
        assert "emailrag_session=" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "Secure" not in set_cookie   # http:// redirect base -> not marked secure

        with connect(DATABASE_URL) as conn:
            user = conn.execute(
                "SELECT id, email FROM users WHERE google_sub = %s", (google_sub,)
            ).fetchone()
            assert user is not None
            user_id, email = user
            assert email == "new-user@example.com"

            token_row = conn.execute(
                "SELECT encrypted_refresh_token FROM oauth_tokens WHERE user_id = %s",
                (user_id,)).fetchone()
            assert token_row is not None
            assert b"r-secret" not in bytes(token_row[0])   # not stored as plaintext

            sync_row = conn.execute(
                "SELECT status FROM sync_state WHERE user_id = %s", (user_id,)).fetchone()
            assert sync_row == ("pending",)

            job = conn.execute(
                "SELECT type, status FROM jobs WHERE user_id = %s", (user_id,)).fetchone()
            assert job == ("initial_sync", "queued")
    finally:
        _cleanup_user(google_sub)


def test_a_returning_user_does_not_get_a_second_initial_sync_job(client, monkeypatch):
    c, settings = client
    google_sub = "sub-returning-user-test"
    try:
        def do_login():
            state = create_state_token(settings.session_secret, "nonce")
            monkeypatch.setattr(routes_module.G, "exchange_code", lambda *a, **k: {
                "refresh_token": "r", "access_token": "a", "expires_in": 3600})
            monkeypatch.setattr(routes_module, "fetch_userinfo", lambda access_token: {
                "sub": google_sub, "email": "returning@example.com"})
            return c.get("/auth/google/callback", params={"code": "abc", "state": state})

        first = do_login()
        assert first.status_code in (302, 307)
        second = do_login()
        assert second.status_code in (302, 307)

        with connect(DATABASE_URL) as conn:
            user_id = conn.execute(
                "SELECT id FROM users WHERE google_sub = %s", (google_sub,)
            ).fetchone()[0]
            jobs = conn.execute(
                "SELECT count(*) FROM jobs WHERE user_id = %s AND type = 'initial_sync'",
                (user_id,)).fetchone()[0]
            assert jobs == 1
    finally:
        _cleanup_user(google_sub)


def test_logout_clears_the_session_cookie(client):
    c, _ = client
    resp = c.post("/auth/google/logout")
    assert resp.status_code in (302, 303)
    set_cookie = resp.headers.get("set-cookie", "")
    assert "emailrag_session=" in set_cookie
