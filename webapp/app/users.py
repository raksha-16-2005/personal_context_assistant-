"""The `users` table: one row per Google account, keyed on the stable `sub`
claim rather than email - an email address can change hands or be reused;
the `sub` claim is Google's permanent identifier for the account.
"""
from __future__ import annotations


def upsert_user(conn, google_sub: str, email: str) -> str:
    """Create the user on first login, or just refresh their email otherwise.

    A single statement rather than select-then-insert: two logins racing on
    the same brand-new account would otherwise both see "no row" and both try
    to insert, and one would fail on the unique constraint instead of the
    other simply winning.
    """
    row = conn.execute(
        """
        INSERT INTO users (google_sub, email) VALUES (%s, %s)
        ON CONFLICT (google_sub) DO UPDATE SET email = EXCLUDED.email
        RETURNING id
        """,
        (google_sub, email),
    ).fetchone()
    return str(row[0])


def delete_user(conn, user_id: str) -> None:
    """The real deletion path: every other table cascades from `users` (see
    schema.sql), so removing this one row is removing all of it - tokens,
    keys, conversations, commitments, calendar suggestions, jobs. Deleting
    the user's `data/index/users/<id>/` directory is the caller's job (it
    isn't in Postgres), see the ingestion module.
    """
    conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


def get_timezone(conn, user_id: str) -> str:
    row = conn.execute(
        "SELECT timezone FROM users WHERE id = %s", (user_id,)).fetchone()
    return row[0] if row else "UTC"


def get_email(conn, user_id: str) -> str:
    row = conn.execute(
        "SELECT email FROM users WHERE id = %s", (user_id,)).fetchone()
    return row[0] if row else ""


def set_timezone(conn, user_id: str, tz_name: str) -> None:
    """Raises `ValueError` for a name `zoneinfo` cannot resolve, before it
    ever reaches the database - a bad zone name should fail the request that
    set it, not silently corrupt every future "today" for this user.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone {tz_name!r}") from exc
    conn.execute("UPDATE users SET timezone = %s WHERE id = %s", (tz_name, user_id))
