from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.jobs.queue import claim_next, complete, enqueue, fail


def test_an_empty_queue_claims_nothing(db_conn):
    db_conn.execute("DELETE FROM jobs")
    assert claim_next(db_conn) is None


def test_enqueue_then_claim_roundtrips_the_payload(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    enqueue(db_conn, "sync_mailbox", user_id=test_user, payload={"full": True})

    job = claim_next(db_conn)
    assert job["type"] == "sync_mailbox"
    assert job["user_id"] == test_user
    assert job["payload"] == {"full": True}
    assert job["attempts"] == 1


def test_a_claimed_job_is_not_claimed_again(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    enqueue(db_conn, "sync_mailbox", user_id=test_user)

    first = claim_next(db_conn)
    assert first is not None
    assert claim_next(db_conn) is None    # already running, not queued


def test_type_filter_only_claims_matching_types(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    enqueue(db_conn, "send_digest", user_id=test_user)
    enqueue(db_conn, "sync_mailbox", user_id=test_user)

    job = claim_next(db_conn, job_types=["sync_mailbox"])
    assert job["type"] == "sync_mailbox"


def test_complete_marks_it_done(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    job_id = enqueue(db_conn, "sync_mailbox", user_id=test_user)
    claim_next(db_conn)
    complete(db_conn, job_id)

    status = db_conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0]
    assert status == "done"


def test_fail_retries_with_backoff_before_giving_up(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    job_id = enqueue(db_conn, "sync_mailbox", user_id=test_user)

    claim_next(db_conn)                      # attempts -> 1
    # retry_after_seconds=0 here so the requeue is immediately reclaimable -
    # the backoff *delay* itself is covered separately below, this test is
    # only about the requeue-vs-give-up decision.
    fail(db_conn, job_id, "boom", retry_after_seconds=0, max_attempts=3)
    row = db_conn.execute("SELECT status, last_error FROM jobs WHERE id = %s",
                          (job_id,)).fetchone()
    assert row == ("queued", "boom")         # requeued, not yet given up

    claim_next(db_conn)                      # attempts -> 2
    fail(db_conn, job_id, "boom again", max_attempts=2)
    status = db_conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0]
    assert status == "failed"                # max_attempts reached


def test_a_job_requeued_after_failure_respects_run_after(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    job_id = enqueue(db_conn, "sync_mailbox", user_id=test_user)
    claim_next(db_conn)
    fail(db_conn, job_id, "boom", retry_after_seconds=3600, max_attempts=5)

    # run_after is an hour out, so a poll right now must not reclaim it.
    assert claim_next(db_conn) is None


def test_deleting_the_user_cascades_to_their_jobs(db_conn, test_user):
    db_conn.execute("DELETE FROM jobs")
    enqueue(db_conn, "sync_mailbox", user_id=test_user)

    db_conn.execute("DELETE FROM users WHERE id = %s", (test_user,))

    remaining = db_conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    assert remaining == 0
