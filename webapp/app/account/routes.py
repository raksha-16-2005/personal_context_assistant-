"""GET /me (who's logged in, for the frontend's own auth check), the Gemini
key settings the frontend's Settings page reads/writes, and DELETE /account
- the real deletion path.

`users.delete_user` removes one row and every other per-user table cascades
from it (tokens, keys, conversations, commitments, calendar suggestions,
digests, jobs - see schema.sql). What cascade cannot reach is this user's
on-disk index directory, since it lives outside Postgres entirely - that is
this route's other job, alongside dropping the now-stale Pipeline out of the
pool's cache and clearing the browser's session cookie so a deleted account
is not still "logged in" client-side.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ..auth.session import COOKIE_NAME
from ..config import Settings
from ..db import connect
from ..deps import get_current_user_id, get_pipeline_pool, get_settings
from ..gemini_keys import delete_gemini_key, delete_gemini_key_2, load_gemini_keys, save_gemini_key
from ..ingestion.worker import user_root
from ..jobs.queue import enqueue
from ..pipeline_pool import PipelinePool
from ..users import delete_user, get_timezone, set_timezone

router = APIRouter(tags=["account"])


class GeminiKeyBody(BaseModel):
    api_key: str
    # A fallback key, used only once every model is quota-exhausted under
    # `api_key` (see llm/client.py's own key-rotation) - optional, most
    # users will only ever set the primary one.
    api_key_2: str | None = None


class TimezoneBody(BaseModel):
    timezone: str


@router.get("/me")
def me(user_id: str = Depends(get_current_user_id),
      settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT email, timezone FROM users WHERE id = %s", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such user")
    return {"id": user_id, "email": row[0], "timezone": row[1]}


FALLBACK_ETA_SECONDS = 60


def _estimate_eta_seconds(status: str, sync_started_at, progress_current: int,
                          progress_total: int) -> int | None:
    """A real projection once there's something to project from - Gmail's
    own exact count for this sync (progress_total, see
    ingestion/worker.py's on_progress wiring) and how long `progress_current`
    of it has actually taken so far (`sync_started_at`). None until then,
    which the caller reports as the flat `FALLBACK_ETA_SECONDS` guess rather
    than a fabricated number with no real signal behind it yet - the first
    poll or two of a sync, before the first `on_progress` callback lands.
    """
    if status not in ("pending", "syncing"):
        return 0
    if not sync_started_at or progress_current <= 0 or progress_total <= 0:
        return None
    elapsed = (datetime.now(timezone.utc) - sync_started_at).total_seconds()
    rate = elapsed / progress_current
    return max(0, round(rate * (progress_total - progress_current)))


@router.get("/sync-status")
def sync_status(user_id: str = Depends(get_current_user_id),
                settings: Settings = Depends(get_settings)):
    """Polled by the frontend while a mailbox is syncing, so a new login
    shows real progress ("41 of 120 messages, about 30s left") instead of
    only ever finding out it's done via a /chat request's 409.

    `eta_seconds` is a genuine projection - (elapsed / progress_current) *
    (progress_total - progress_current), where progress_total is Gmail's own
    exact message count for this sync's query (see ingestion/worker.py's
    on_progress wiring), not an estimate - once there's enough progress to
    project from. Before that (the first poll or two of a sync), it falls
    back to FALLBACK_ETA_SECONDS rather than a number with no real signal
    behind it.
    """
    with connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT status, messages_seen, full_history_synced, error_detail, "
            "sync_started_at, progress_current, progress_total "
            "FROM sync_state WHERE user_id = %s", (user_id,)).fetchone()
    # No row yet is effectively "pending" - auth/routes.py's callback
    # inserts one synchronously before the frontend ever loads, so this only
    # covers a request that somehow raced that insert.
    (status, messages_seen, full_history_synced, error_detail, sync_started_at,
     progress_current, progress_total) = row or ("pending", 0, False, "", None, 0, 0)

    eta_seconds = _estimate_eta_seconds(
        status, sync_started_at, progress_current, progress_total)
    return {
        "status": status,
        "messages_seen": messages_seen,
        "full_history_synced": full_history_synced,
        "error_detail": error_detail,
        "progress_current": progress_current,
        "progress_total": progress_total,
        "eta_seconds": FALLBACK_ETA_SECONDS if eta_seconds is None else eta_seconds,
        # True only once eta_seconds reflects this user's own real fetch,
        # not the flat fallback - the frontend uses this to say "about" vs
        # state a number with more confidence than it deserves.
        "eta_is_estimate": eta_seconds is None and status in ("pending", "syncing"),
    }


@router.put("/account/timezone", status_code=204)
def update_timezone(body: TimezoneBody, user_id: str = Depends(get_current_user_id),
                    settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        try:
            set_timezone(conn, user_id, body.timezone)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


@router.get("/account/gemini-key")
def gemini_key_status(user_id: str = Depends(get_current_user_id),
                      settings: Settings = Depends(get_settings)):
    # Never returns the keys themselves, only whether each is on file - the
    # same "decrypted only long enough to use it" boundary gemini_keys.py
    # already draws for /chat applies just as much to the browser as to any
    # other caller.
    with connect(settings.database_url) as conn:
        keys = load_gemini_keys(conn, user_id, settings.master_key)
    return {"has_key": len(keys) >= 1, "has_key_2": len(keys) >= 2}


@router.put("/account/gemini-key", status_code=204)
def set_gemini_key(body: GeminiKeyBody, user_id: str = Depends(get_current_user_id),
                   settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        save_gemini_key(conn, user_id, settings.master_key, body.api_key, body.api_key_2)
        # Extraction only ever ran automatically right after a sync (see
        # jobs/runner.py's `_handle_sync`) - a user who pastes their key
        # *after* their first sync already finished would otherwise never
        # get commitments/calendar suggestions until their next sync, which
        # for an incremental sync only happens when new mail arrives.
        # `extract_for_user` no-ops cheaply if nothing has synced yet, so
        # this is safe to enqueue unconditionally rather than checking first.
        enqueue(conn, "extract_commitments", user_id=user_id)


@router.delete("/account/gemini-key", status_code=204)
def remove_gemini_key(user_id: str = Depends(get_current_user_id),
                      settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        delete_gemini_key(conn, user_id)


@router.delete("/account/gemini-key-2", status_code=204)
def remove_gemini_key_2(user_id: str = Depends(get_current_user_id),
                        settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        delete_gemini_key_2(conn, user_id)


@router.delete("/account", status_code=204)
def delete_account(user_id: str = Depends(get_current_user_id),
                   settings: Settings = Depends(get_settings),
                   pool: PipelinePool = Depends(get_pipeline_pool)):
    with connect(settings.database_url) as conn:
        delete_user(conn, user_id)

    # Dropped from the cache before the directory goes, not after: a request
    # racing this one must never be handed a Pipeline whose files are mid-delete.
    pool.invalidate(user_id)
    shutil.rmtree(user_root(settings.user_index_root, user_id), ignore_errors=True)

    response = Response(status_code=204)
    response.delete_cookie(COOKIE_NAME)
    return response
