from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.commitments import (generate_calendar_suggestions, insert_commitments,
                             load_commitments_for_router)
from emailrag.extraction.schema import Commitment
from emailrag.router.sql import filter_commitments, parse_window


def _commitment(**overrides) -> Commitment:
    base = dict(message_id="m1", text="send the agenda", kind="deliverable",
               direction="i_owe", owner="alice@example.com",
               counterparty="bob@example.com", due_phrase="Friday",
               due_at=date(2026, 8, 14), model="gemini-2.5-flash")
    base.update(overrides)
    return Commitment(**base)


def test_insert_commitments_stores_a_new_row(db_conn, test_user):
    inserted = insert_commitments(db_conn, test_user, [_commitment()])
    assert inserted == 1

    row = db_conn.execute(
        "SELECT text, owner, due_at FROM commitments WHERE user_id = %s", (test_user,)
    ).fetchone()
    assert row == ("send the agenda", "alice@example.com", date(2026, 8, 14))


def test_insert_commitments_is_idempotent(db_conn, test_user):
    insert_commitments(db_conn, test_user, [_commitment()])
    second = insert_commitments(db_conn, test_user, [_commitment()])   # same run again

    assert second == 0
    count = db_conn.execute(
        "SELECT count(*) FROM commitments WHERE user_id = %s", (test_user,)
    ).fetchone()[0]
    assert count == 1


def test_insert_commitments_is_idempotent_across_a_different_fallback_model(
        db_conn, test_user):
    # The bug this guards against: Gemini's automatic quota-rotation
    # fallback (llm/client.py's GEMINI_FALLBACKS) means the same message can
    # get re-extracted under a *different* model name on a later rescan.
    # `model` used to be part of the uniqueness key, so this produced two
    # rows for one real commitment instead of one - visible in production as
    # duplicate-looking citations for the same deadline.
    insert_commitments(db_conn, test_user, [_commitment(model="gemini-2.5-flash")])
    second = insert_commitments(db_conn, test_user, [_commitment(model="gemini-3.5-flash")])

    assert second == 0
    count = db_conn.execute(
        "SELECT count(*) FROM commitments WHERE user_id = %s", (test_user,)
    ).fetchone()[0]
    assert count == 1


def test_insert_commitments_keeps_two_users_separate(db_conn, test_user):
    other = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        ("commitments-test-other-sub", "other@example.com")).fetchone()[0]
    try:
        insert_commitments(db_conn, test_user, [_commitment()])
        insert_commitments(db_conn, str(other), [_commitment()])   # same message/model/text

        count = db_conn.execute(
            "SELECT count(*) FROM commitments WHERE user_id IN (%s, %s)",
            (test_user, other)).fetchone()[0]
        assert count == 2
    finally:
        db_conn.execute("DELETE FROM users WHERE id = %s", (other,))


def test_generate_calendar_suggestions_only_covers_dated_commitments(db_conn, test_user):
    insert_commitments(db_conn, test_user, [
        _commitment(text="dated one", due_at=date(2026, 8, 14)),
        _commitment(text="no date", due_phrase="", due_at=None),
    ])

    created = generate_calendar_suggestions(db_conn, test_user)

    assert created == 1
    rows = db_conn.execute(
        "SELECT status FROM calendar_suggestions WHERE user_id = %s", (test_user,)
    ).fetchall()
    assert rows == [("pending",)]


def test_generate_calendar_suggestions_is_idempotent(db_conn, test_user):
    insert_commitments(db_conn, test_user, [_commitment()])
    generate_calendar_suggestions(db_conn, test_user)
    second = generate_calendar_suggestions(db_conn, test_user)

    assert second == 0
    count = db_conn.execute(
        "SELECT count(*) FROM calendar_suggestions WHERE user_id = %s", (test_user,)
    ).fetchone()[0]
    assert count == 1


def test_load_commitments_for_router_returns_real_commitment_rows(db_conn, test_user):
    insert_commitments(db_conn, test_user, [_commitment()])

    loaded = load_commitments_for_router(db_conn, test_user)

    assert len(loaded) == 1
    assert isinstance(loaded[0], Commitment)
    assert loaded[0].text == "send the agenda"
    assert loaded[0].due_at == date(2026, 8, 14)


def test_load_commitments_for_router_only_returns_this_users_rows(db_conn, test_user):
    other = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        ("router-test-other-sub", "other@example.com")).fetchone()[0]
    try:
        insert_commitments(db_conn, test_user, [_commitment(text="mine")])
        insert_commitments(db_conn, str(other), [_commitment(text="theirs")])

        loaded = load_commitments_for_router(db_conn, test_user)

        assert [c.text for c in loaded] == ["mine"]
    finally:
        db_conn.execute("DELETE FROM users WHERE id = %s", (other,))


def test_loaded_commitments_work_with_the_routers_own_temporal_filter(db_conn, test_user):
    # The point of load_commitments_for_router: what it returns has to be
    # usable by router/sql.py's in-memory filter unchanged, not almost the
    # right shape.
    insert_commitments(db_conn, test_user, [
        _commitment(text="due this week", due_at=date(2026, 8, 14)),
        _commitment(text="due next month", due_at=date(2026, 9, 20)),
    ])
    loaded = load_commitments_for_router(db_conn, test_user)

    window = parse_window("what's due this week", as_of=date(2026, 8, 10))
    hits = filter_commitments(loaded, window)

    assert [c.text for c in hits] == ["due this week"]
