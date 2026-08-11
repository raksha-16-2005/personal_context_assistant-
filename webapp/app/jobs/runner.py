"""Poll the job queue and dispatch each claimed job to its handler.

Run as its own process (`python -m app.jobs.runner`), separate from the web
process that serves /chat - a slow or stuck sync job must never block a live
request, and a burst of /chat traffic must never starve queued syncs. They
only share the database and, in production, the same PipelinePool instance
if wired to one (so a completed sync can invalidate that user's cached
Pipeline - see main.py for where that wiring belongs once this runs
alongside the web process rather than as a separate deployment).
"""
from __future__ import annotations

import time

from . import queue

HANDLERS: dict = {}


def register(job_type: str):
    def deco(fn):
        HANDLERS[job_type] = fn
        return fn
    return deco


@register("initial_sync")
def _handle_initial_sync(conn, settings, job: dict) -> dict:
    from ..ingestion.worker import RECENT_SYNC_DAYS, sync_user
    from emailrag.corpus.gmail import since_query

    # Recent-only, on purpose - see RECENT_SYNC_DAYS's own docstring. A new
    # user reaches "ready" in about a minute instead of waiting for their
    # entire history; `backfill_history` (enqueued below) fills in the rest
    # in the background.
    result = sync_user(conn, settings, job["user_id"], query=since_query(RECENT_SYNC_DAYS))
    queue.enqueue(conn, "extract_commitments", user_id=job["user_id"])
    queue.enqueue(conn, "backfill_history", user_id=job["user_id"])
    return result


@register("incremental_sync")
def _handle_incremental_sync(conn, settings, job: dict) -> dict:
    from ..ingestion.worker import sync_user
    result = sync_user(conn, settings, job["user_id"])
    # Chained rather than left to a separate schedule: a sync just changed
    # what mail exists to extract from, so the next extraction pass should
    # see it on the next poll, not wait for an unrelated trigger.
    queue.enqueue(conn, "extract_commitments", user_id=job["user_id"])
    return result


@register("backfill_history")
def _handle_backfill_history(conn, settings, job: dict) -> dict:
    from ..ingestion.worker import backfill_history
    return backfill_history(conn, settings, job["user_id"])


@register("extract_commitments")
def _handle_extract_commitments(conn, settings, job: dict) -> dict:
    from ..extraction.worker import extract_for_user
    return extract_for_user(conn, settings, job["user_id"])


@register("generate_digest")
def _handle_generate_digest(conn, settings, job: dict) -> dict:
    from ..digest.service import generate_digest
    return generate_digest(conn, settings, job["user_id"])


def run_once(conn, settings) -> bool:
    """Claim and process at most one job. Returns False if the queue had
    nothing due - the caller's cue to back off rather than spin."""
    job = queue.claim_next(conn)
    if job is None:
        return False

    handler = HANDLERS.get(job["type"])
    if handler is None:
        queue.fail(conn, job["id"], f"no handler registered for job type {job['type']!r}",
                  max_attempts=1)
        return True

    try:
        handler(conn, settings, job)
        queue.complete(conn, job["id"])
    except Exception as exc:                                      # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        queue.fail(conn, job["id"], detail)
        # A sync failure has to be visible to the user, not just to whoever
        # reads the jobs table - sync_state.status is what /chat and the UI's
        # "still syncing" check actually read.
        if job["type"] in ("initial_sync", "incremental_sync") and job["user_id"]:
            conn.execute(
                "UPDATE sync_state SET status = 'error', error_detail = %s "
                "WHERE user_id = %s", (detail, job["user_id"]))
    return True


def main_loop(poll_interval_seconds: float = 5.0) -> None:
    from ..config import load_settings
    from ..db import connect
    from ..digest.service import schedule_due_digests
    from ..ingestion.worker import schedule_due_syncs

    settings = load_settings()
    with connect(settings.database_url) as conn:
        while True:
            # Cheap enough to call every pass (an indexed lookup on a tiny
            # table) that it doesn't need its own timer separate from the
            # job poll it already rides along with - revisit if that stops
            # being true, matching this queue's own "simplest thing that
            # rate-limits work" design choice.
            schedule_due_digests(conn)
            schedule_due_syncs(conn)
            if not run_once(conn, settings):
                time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    main_loop()
