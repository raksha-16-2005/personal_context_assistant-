from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.security import DecryptionError, assert_columns_encrypted, decrypt, encrypt, looks_encrypted

KEY = Fernet.generate_key().decode()


def test_roundtrip():
    ciphertext = encrypt(KEY, "a-refresh-token")
    assert decrypt(KEY, ciphertext) == "a-refresh-token"


def test_ciphertext_is_not_the_plaintext_bytes():
    # The whole point: what lands on disk must not simply be the secret.
    ciphertext = encrypt(KEY, "a-refresh-token")
    assert b"a-refresh-token" not in ciphertext


def test_wrong_key_cannot_decrypt():
    other_key = Fernet.generate_key().decode()
    ciphertext = encrypt(KEY, "a-refresh-token")
    with pytest.raises(DecryptionError):
        decrypt(other_key, ciphertext)


def test_looks_encrypted_rejects_plaintext():
    # A bug that stored the raw secret instead of calling encrypt() must be
    # catchable by inspecting the actual bytes, not by trusting that encrypt()
    # was called.
    assert not looks_encrypted(KEY, b"a-refresh-token")


def test_looks_encrypted_accepts_real_ciphertext():
    assert looks_encrypted(KEY, encrypt(KEY, "a-refresh-token"))


class _FakeConn:
    """Just enough of psycopg's Connection interface for assert_columns_encrypted."""

    def __init__(self, rows_by_query: dict[str, list[tuple]]):
        self._rows_by_query = rows_by_query

    def execute(self, sql: str):
        for key, rows in self._rows_by_query.items():
            if key in sql:
                return _FakeCursor(rows)
        return _FakeCursor([])


class _FakeCursor:
    def __init__(self, rows: list[tuple]):
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_assert_columns_encrypted_passes_on_real_ciphertext():
    conn = _FakeConn({
        "oauth_tokens": [("user-1", encrypt(KEY, "refresh"))],
        "gemini_keys": [("user-1", encrypt(KEY, "gemini-key"))],
    })
    assert_columns_encrypted(conn, KEY)   # must not raise


def test_assert_columns_encrypted_catches_a_plaintext_leak():
    # This is the regression this function exists to catch: a code path that
    # bypassed encrypt() and wrote the secret straight to a bytea column.
    conn = _FakeConn({
        "oauth_tokens": [("user-1", b"a-raw-refresh-token")],
        "gemini_keys": [],
    })
    with pytest.raises(RuntimeError, match="plaintext secret"):
        assert_columns_encrypted(conn, KEY)
