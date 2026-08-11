from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.commitments import generate_calendar_suggestions, insert_commitments
from app.config import Settings
from app.digest.service import (due_users, generate_digest, get_latest, get_settings_row,
                                save_settings, schedule_due_digests)
from emailrag.extraction.schema import Commitment


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


def test_default_settings_are_disabled_before_anything_is_saved(db_conn, test_user):
    assert get_settings_row(db_conn, test_user) == {
        "enabled": False, "send_hour_utc": 13, "last_sent_utc": None}


def test_save_settings_then_read_them_back(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=7)
    row = get_settings_row(db_conn, test_user)
    assert row["enabled"] is True
    assert row["send_hour_utc"] == 7
    assert row["last_sent_utc"] is None


def test_save_settings_upserts_rather_than_duplicating(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=7)
    save_settings(db_conn, test_user, enabled=False, send_hour_utc=20)

    count = db_conn.execute(
        "SELECT count(*) FROM digest_settings WHERE user_id = %s", (test_user,)
    ).fetchone()[0]
    assert count == 1
    assert get_settings_row(db_conn, test_user) == {
        "enabled": False, "send_hour_utc": 20, "last_sent_utc": None}


def test_generate_digest_includes_due_soon_commitments_and_pending_suggestions(
        db_conn, test_user, tmp_path):
    # `test_user` gets `timezone`'s schema default (UTC, see schema.sql), so
    # "today" has to be anchored in UTC here too - `_due_soon` resolves the
    # user's "today" from their stored zone, not the test machine's local
    # clock, and those can disagree for several hours a day.
    settings = _settings(tmp_path)
    insert_commitments(db_conn, test_user, [Commitment(
        message_id="m1", text="send the agenda",
        due_at=datetime.now(timezone.utc).date() + timedelta(days=2),
        model="gemini-2.5-flash")])
    generate_calendar_suggestions(db_conn, test_user)

    digest = generate_digest(db_conn, settings, test_user)

    assert digest["pending_calendar_suggestions"] == 1
    assert len(digest["due_soon"]) == 1
    assert digest["due_soon"][0]["text"] == "send the agenda"
    assert digest["new_messages"] == 0    # no messages.parquet on disk for this user


def test_generate_digest_excludes_commitments_outside_the_lookahead_window(
        db_conn, test_user, tmp_path):
    settings = _settings(tmp_path)
    insert_commitments(db_conn, test_user, [Commitment(
        message_id="m1", text="far future thing",
        due_at=datetime.now(timezone.utc).date() + timedelta(days=30),
        model="gemini-2.5-flash")])

    digest = generate_digest(db_conn, settings, test_user)
    assert digest["due_soon"] == []


def test_generate_digest_persists_and_get_latest_returns_it(db_conn, test_user, tmp_path):
    settings = _settings(tmp_path)
    first = generate_digest(db_conn, settings, test_user)
    latest = get_latest(db_conn, test_user)

    assert latest["id"] == first["id"]
    assert latest["due_soon"] == []


def test_get_latest_is_none_when_nothing_generated_yet(db_conn, test_user):
    assert get_latest(db_conn, test_user) is None


def test_due_users_excludes_disabled_users(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=False, send_hour_utc=13)
    db_conn.execute(
        "UPDATE digest_settings SET send_hour_utc = "
        "extract(hour from now() at time zone 'utc') WHERE user_id = %s", (test_user,))
    assert str(test_user) not in due_users(db_conn)


def test_due_users_excludes_a_wrong_hour(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=(_current_utc_hour() + 5) % 24)
    assert str(test_user) not in due_users(db_conn)


def test_due_users_includes_an_enabled_user_at_the_right_hour_never_sent(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=_current_utc_hour())
    assert str(test_user) in due_users(db_conn)


def test_due_users_excludes_a_user_already_sent_today(db_conn, test_user):
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=_current_utc_hour())
    db_conn.execute(
        "UPDATE digest_settings SET last_sent_utc = now() WHERE user_id = %s", (test_user,))
    assert str(test_user) not in due_users(db_conn)


def test_schedule_due_digests_enqueues_and_marks_sent(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    save_settings(db_conn, test_user, enabled=True, send_hour_utc=_current_utc_hour())

    scheduled = schedule_due_digests(db_conn)

    assert str(test_user) in scheduled
    job = db_conn.execute(
        "SELECT type, status FROM jobs WHERE user_id = %s", (test_user,)).fetchone()
    assert job == ("generate_digest", "queued")

    # Immediately marked sent, so a poll moments later does not enqueue twice.
    again = schedule_due_digests(db_conn)
    assert str(test_user) not in again


def _current_utc_hour() -> int:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).hour
