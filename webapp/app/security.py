"""Envelope encryption for oauth_tokens and gemini_keys.

Both columns hold secrets that can read someone's entire mailbox or spend
someone's own API quota, so they are never written in plaintext - not
"protected by access control", the bytes on disk are ciphertext. This is the
direct successor to `scripts/check_privacy.py`'s principle of checking the
actual artifact rather than trusting a rule that says it shouldn't happen -
applied to a running database instead of a git repo, see `looks_encrypted`
and `assert_columns_encrypted` below.

One Fernet key (MASTER_KEY), from the hosting platform's secret manager, never
from this database and never logged. Losing it makes every stored token/key
permanently undecryptable - that is the correct failure mode for a key you
never want recoverable by anyone who only has the database.
"""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken


class DecryptionError(RuntimeError):
    pass


def encrypt(master_key: str, plaintext: str) -> bytes:
    return Fernet(master_key).encrypt(plaintext.encode())


def decrypt(master_key: str, ciphertext: bytes) -> str:
    try:
        return Fernet(master_key).decrypt(ciphertext).decode()
    except InvalidToken as exc:
        raise DecryptionError(
            "could not decrypt - wrong MASTER_KEY, or this value was never "
            "Fernet-encrypted in the first place") from exc


def looks_encrypted(master_key: str, value: bytes) -> bool:
    """True only if `value` actually decrypts under this key.

    Used as a runtime assertion over the real column contents, not a naming
    convention: a column called `encrypted_*` that happens to hold plaintext
    because a bug bypassed `encrypt()` is exactly the failure this has to
    catch, and it cannot be caught by trusting the column name.
    """
    try:
        Fernet(master_key).decrypt(value)
        return True
    except InvalidToken:
        return False


def assert_columns_encrypted(conn, master_key: str) -> None:
    """Raise if any stored token/key row is not real ciphertext.

    Intended to run at app startup and in CI against a seeded test database -
    the same "fail loudly before this reaches production" spirit as
    `check_privacy.py`'s pre-push hook, just checking a database instead of a
    git tree.
    """
    problems: list[str] = []
    for table, column in (("oauth_tokens", "encrypted_refresh_token"),
                          ("oauth_tokens", "encrypted_access_token"),
                          ("gemini_keys", "encrypted_key")):
        rows = conn.execute(f"SELECT user_id, {column} FROM {table}").fetchall()
        for user_id, value in rows:
            if not looks_encrypted(master_key, bytes(value)):
                problems.append(f"{table}.{column} for user {user_id} is not "
                                f"valid ciphertext under MASTER_KEY")
    if problems:
        raise RuntimeError(
            "refusing to start: plaintext secret(s) found in the database:\n  "
            + "\n  ".join(problems))


if __name__ == "__main__":
    # `python -m app.security` - a deploy-time gate (Fly's `release_command`,
    # see ../fly.toml), not a request-time or test-time one. Every pytest run
    # deliberately encrypts under a fresh, random key per test for isolation
    # (see tests/conftest.py), so checking "does everything in the table
    # decrypt under *the one* production MASTER_KEY" only makes sense against
    # the one real deployment - running it from the app's own lifespan would
    # fail on every test's own leftover rows, encrypted under a different key.
    from .config import load_settings
    from .db import connect

    settings = load_settings()
    with connect(settings.database_url) as conn:
        assert_columns_encrypted(conn, settings.master_key)
    print("all stored tokens/keys are valid ciphertext under MASTER_KEY")
