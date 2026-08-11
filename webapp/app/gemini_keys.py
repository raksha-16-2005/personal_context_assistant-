"""The user's own pasted Gemini API key - encrypted at rest, decrypted only
long enough to build an `LLM` for that user's own `/chat` request. Never
logged, never sent anywhere but back to Google under that user's own quota.
"""
from __future__ import annotations

from .security import decrypt, encrypt


def save_gemini_key(conn, user_id: str, master_key: str, api_key: str) -> None:
    encrypted = encrypt(master_key, api_key)
    conn.execute(
        """
        INSERT INTO gemini_keys (user_id, encrypted_key) VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            encrypted_key = EXCLUDED.encrypted_key, updated_at = now()
        """,
        (user_id, encrypted),
    )


def load_gemini_key(conn, user_id: str, master_key: str) -> str | None:
    """None means "no key saved yet", distinct from an empty string - the
    caller (the /chat route) needs to tell "please paste a key" apart from
    any other failure.
    """
    row = conn.execute(
        "SELECT encrypted_key FROM gemini_keys WHERE user_id = %s", (user_id,)
    ).fetchone()
    if row is None:
        return None
    return decrypt(master_key, bytes(row[0]))


def delete_gemini_key(conn, user_id: str) -> None:
    conn.execute("DELETE FROM gemini_keys WHERE user_id = %s", (user_id,))
