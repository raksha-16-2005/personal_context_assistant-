from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.jobs import queue, runner


def test_an_empty_queue_returns_false(db_conn):
    db_conn.execute("DELETE FROM jobs")
    assert runner.run_once(db_conn, settings=None) is False


def test_a_registered_handler_is_dispatched_and_the_job_completes(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    calls = []
    runner.HANDLERS["test_job"] = lambda conn, settings, job: calls.append(job) or None
    try:
        job_id = queue.enqueue(db_conn, "test_job", user_id=test_user, payload={"x": 1})
        processed = runner.run_once(db_conn, settings=None)

        assert processed is True
        assert len(calls) == 1
        assert calls[0]["payload"] == {"x": 1}
        status = db_conn.execute(
            "SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0]
        assert status == "done"
    finally:
        del runner.HANDLERS["test_job"]


def test_an_unregistered_job_type_fails_without_retrying(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    job_id = queue.enqueue(db_conn, "no_such_handler", user_id=test_user)

    runner.run_once(db_conn, settings=None)

    row = db_conn.execute(
        "SELECT status, last_error FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row[0] == "failed"
    assert "no handler registered" in row[1]


def test_a_failing_sync_job_marks_sync_state_as_error(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'syncing')",
                    (test_user,))

    def boom(conn, settings, job):
        raise RuntimeError("gmail token expired")

    runner.HANDLERS["initial_sync"], original = boom, runner.HANDLERS["initial_sync"]
    try:
        queue.enqueue(db_conn, "initial_sync", user_id=test_user)
        runner.run_once(db_conn, settings=None)

        row = db_conn.execute(
            "SELECT status, error_detail FROM sync_state WHERE user_id = %s",
            (test_user,)).fetchone()
        assert row[0] == "error"
        assert "gmail token expired" in row[1]
    finally:
        runner.HANDLERS["initial_sync"] = original


def test_initial_sync_chains_extraction_and_backfill(db_conn, test_user, monkeypatch):
    db_conn.execute("DELETE FROM jobs")
    calls = []
    monkeypatch.setattr(
        "app.ingestion.worker.sync_user",
        lambda conn, settings, user_id, query="": calls.append(query) or {"ok": True})

    queue.enqueue(db_conn, "initial_sync", user_id=test_user)
    runner.run_once(db_conn, settings=None)

    assert len(calls) == 1
    assert calls[0].startswith("after:")          # RECENT_SYNC_DAYS's since_query

    queued_types = {r[0] for r in db_conn.execute(
        "SELECT type FROM jobs WHERE user_id = %s AND status = 'queued'",
        (test_user,)).fetchall()}
    assert queued_types == {"extract_commitments", "backfill_history"}


def test_a_failing_non_sync_job_does_not_touch_sync_state(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    db_conn.execute("INSERT INTO sync_state (user_id, status) VALUES (%s, 'ready')",
                    (test_user,))
    runner.HANDLERS["test_job"] = lambda conn, settings, job: (_ for _ in ()).throw(
        RuntimeError("boom"))
    try:
        queue.enqueue(db_conn, "test_job", user_id=test_user)
        runner.run_once(db_conn, settings=None)

        status = db_conn.execute(
            "SELECT status FROM sync_state WHERE user_id = %s", (test_user,)).fetchone()[0]
        assert status == "ready"          # untouched
    finally:
        del runner.HANDLERS["test_job"]
