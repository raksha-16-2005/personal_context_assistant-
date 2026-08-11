from __future__ import annotations

import sys
import time
from pathlib import Path

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.security import looks_encrypted
from app.tokens.store import DbTokenStore
from emailrag.corpus.gmail import GmailClient, NeedsAuthorization

KEY = Fernet.generate_key().decode()


def _store(db_conn, user_id, data=None):
    store = DbTokenStore(conn=db_conn, user_id=user_id, master_key=KEY)
    if data is not None:
        store.data = data
    return store


def test_tokens_roundtrip_through_postgres(db_conn, test_user):
    store = _store(db_conn, test_user, {
        "refresh_token": "r-secret", "access_token": "a-secret",
        "scope": "gmail.readonly calendar.events",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    })
    store.save()

    reloaded = _store(db_conn, test_user).load()
    assert reloaded.refresh_token == "r-secret"
    assert reloaded.access_token == "a-secret"
    assert reloaded.data["scope"] == "gmail.readonly calendar.events"


def test_the_stored_bytes_are_ciphertext_not_the_secret(db_conn, test_user):
    _store(db_conn, test_user, {
        "refresh_token": "r-secret", "access_token": "a-secret",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()

    enc_refresh, enc_access = db_conn.execute(
        "SELECT encrypted_refresh_token, encrypted_access_token "
        "FROM oauth_tokens WHERE user_id = %s", (test_user,),
    ).fetchone()
    assert b"r-secret" not in bytes(enc_refresh)
    assert looks_encrypted(KEY, bytes(enc_refresh))
    assert looks_encrypted(KEY, bytes(enc_access))


def test_a_missing_row_is_not_an_error(db_conn, test_user):
    store = _store(db_conn, test_user).load()
    assert store.refresh_token == ""


def test_saving_twice_upserts_rather_than_duplicating(db_conn, test_user):
    _store(db_conn, test_user, {
        "refresh_token": "first", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()
    _store(db_conn, test_user, {
        "refresh_token": "second", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()

    rows = db_conn.execute(
        "SELECT count(*) FROM oauth_tokens WHERE user_id = %s", (test_user,)
    ).fetchone()
    assert rows[0] == 1
    assert _store(db_conn, test_user).load().refresh_token == "second"


def test_deleting_the_user_cascades_to_their_tokens(db_conn, test_user):
    _store(db_conn, test_user, {
        "refresh_token": "r", "access_token": "a",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()

    db_conn.execute("DELETE FROM users WHERE id = %s", (test_user,))

    remaining = db_conn.execute(
        "SELECT count(*) FROM oauth_tokens WHERE user_id = %s", (test_user,)
    ).fetchone()
    assert remaining[0] == 0


def test_gmail_client_accepts_a_db_token_store_unmodified(db_conn, test_user):
    # The whole point of splitting TokenStoreBase out of TokenStore: GmailClient
    # must not need to know or care which backend it was handed.
    _store(db_conn, test_user, {
        "refresh_token": "r", "access_token": "a-valid-token",
        "expires_at": time.time() + 3600, "issued_at": time.time(),
    }).save()

    # GmailClient.__init__ calls .load() itself - a fresh, unsaved store here
    # is enough; it is expected to re-fetch from Postgres.
    client = GmailClient("cid", "secret", tokens=_store(db_conn, test_user))
    assert client._bearer() == "a-valid-token"


def test_gmail_client_reports_how_to_reauthorize_with_no_refresh_token(db_conn, test_user):
    store = _store(db_conn, test_user, {}).load()
    client = GmailClient("cid", "secret", tokens=store)
    try:
        client._bearer()
        assert False, "expected NeedsAuthorization"
    except NeedsAuthorization as exc:
        assert "gmail_auth" in str(exc) or "refresh token" in str(exc)
