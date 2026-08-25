"""The user's own pasted Gemini API key(s) - encrypted at rest, decrypted
only long enough to build an `LLM` for that user's own request. Never
logged, never sent anywhere but back to Google under that user's own quota.

A second, optional key exists purely as a fallback: Gemini's free-tier quota
is metered per key as well as per model, so once every model is exhausted
under the first key (see llm/client.py's own key-rotation), a second key
still has its own untouched quota to answer from instead of failing the
request outright.
"""
from __future__ import annotations

from .security import decrypt, encrypt


def save_gemini_key(conn, user_id: str, master_key: str, api_key: str,
                    api_key_2: str | None = None) -> None:
    """`api_key_2` omitted (None) means "leave whatever backup key is
    already saved alone", not "clear it" - a key is never echoed back to the
    browser once saved (see the module docstring), so a save that only
    means to update the primary key has no way to resend a backup it never
    had in the first place. Actually clearing a saved backup key is
    `delete_gemini_key_2`'s job, a separate, explicit action.
    """
    encrypted = encrypt(master_key, api_key)
    encrypted_2 = encrypt(master_key, api_key_2) if api_key_2 else None
    conn.execute(
        """
        INSERT INTO gemini_keys (user_id, encrypted_key, encrypted_key_2)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            encrypted_key = EXCLUDED.encrypted_key,
            encrypted_key_2 = COALESCE(EXCLUDED.encrypted_key_2, gemini_keys.encrypted_key_2),
            updated_at = now()
        """,
        (user_id, encrypted, encrypted_2),
    )


def load_gemini_keys(conn, user_id: str, master_key: str) -> list[str]:
    """This user's saved keys, primary first. Empty list means "no key saved
    yet" - callers that need to tell that apart from any other failure
    should check emptiness, the same boundary the old single-key
    `load_gemini_key` drew with `None`.
    """
    row = conn.execute(
        "SELECT encrypted_key, encrypted_key_2 FROM gemini_keys WHERE user_id = %s",
        (user_id,)).fetchone()
    if row is None:
        return []
    keys = [decrypt(master_key, bytes(row[0]))]
    if row[1] is not None:
        keys.append(decrypt(master_key, bytes(row[1])))
    return keys


def delete_gemini_key(conn, user_id: str) -> None:
    conn.execute("DELETE FROM gemini_keys WHERE user_id = %s", (user_id,))


def delete_gemini_key_2(conn, user_id: str) -> None:
    """Clears just the backup key, leaving the primary (and the row itself)
    alone - the counterpart to save_gemini_key's own COALESCE behavior,
    which otherwise has no way to remove a backup key once one is set."""
    conn.execute(
        "UPDATE gemini_keys SET encrypted_key_2 = NULL, updated_at = now() "
        "WHERE user_id = %s", (user_id,))
