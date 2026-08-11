"""A DB-polled job queue: the simplest thing that rate-limits concurrent
ingestion/digest work, not a permanent architecture choice - see the plan's
"Revisit later" section for when this stops being enough (Celery/RQ + Redis,
if concurrent syncs start queuing longer than users will tolerate). What it
has to get right from day one is not letting two worker processes claim the
same job, which `FOR UPDATE SKIP LOCKED` guarantees without either one
blocking on the other's row.
"""
from __future__ import annotations

import json


def enqueue(conn, job_type: str, user_id: str | None = None,
           payload: dict | None = None) -> int:
    row = conn.execute(
        "INSERT INTO jobs (type, user_id, payload) VALUES (%s, %s, %s) RETURNING id",
        (job_type, user_id, json.dumps(payload or {})),
    ).fetchone()
    return row[0]


def claim_next(conn, job_types: list[str] | None = None) -> dict | None:
    """Atomically claim the oldest due, queued job, or None if there isn't one.

    The scalar-subquery `WHERE id = (... FOR UPDATE SKIP LOCKED LIMIT 1)` form
    is what makes this safe with more than one worker polling the same table:
    two workers racing on the same row never both claim it, and neither
    blocks waiting on the other's row - it just moves on to the next one.
    """
    type_filter = "AND type = ANY(%(types)s)" if job_types else ""
    row = conn.execute(
        f"""
        UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = now()
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'queued' AND run_after <= now() {type_filter}
            ORDER BY run_after
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, type, user_id, payload, attempts
        """,
        {"types": job_types} if job_types else {},
    ).fetchone()
    if row is None:
        return None
    job_id, job_type, user_id, payload, attempts = row
    return {"id": job_id, "type": job_type,
            "user_id": str(user_id) if user_id else None,
            "payload": payload, "attempts": attempts}


def complete(conn, job_id: int) -> None:
    conn.execute("UPDATE jobs SET status = 'done', updated_at = now() WHERE id = %s",
                (job_id,))


def fail(conn, job_id: int, error: str, retry_after_seconds: int = 60,
        max_attempts: int = 3) -> None:
    """Retry with linear backoff up to `max_attempts`, then leave it `failed`
    for a human to look at rather than retrying forever."""
    row = conn.execute("SELECT attempts FROM jobs WHERE id = %s", (job_id,)).fetchone()
    attempts = row[0] if row else max_attempts
    if attempts >= max_attempts:
        conn.execute(
            "UPDATE jobs SET status = 'failed', last_error = %s, updated_at = now() "
            "WHERE id = %s", (error, job_id))
    else:
        conn.execute(
            "UPDATE jobs SET status = 'queued', last_error = %s, "
            "run_after = now() + (%s * interval '1 second'), updated_at = now() "
            "WHERE id = %s", (error, retry_after_seconds * attempts, job_id))
