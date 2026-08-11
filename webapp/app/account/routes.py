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

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ..auth.session import COOKIE_NAME
from ..config import Settings
from ..db import connect
from ..deps import get_current_user_id, get_pipeline_pool, get_settings
from ..gemini_keys import delete_gemini_key, load_gemini_key, save_gemini_key
from ..ingestion.worker import user_root
from ..jobs.queue import enqueue
from ..pipeline_pool import PipelinePool
from ..users import delete_user, get_timezone, set_timezone

router = APIRouter(tags=["account"])


class GeminiKeyBody(BaseModel):
    api_key: str


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
    # Never returns the key itself, only whether one is on file - the same
    # "decrypted only long enough to use it" boundary gemini_keys.py already
    # draws for /chat applies just as much to the browser as to any other
    # caller.
    with connect(settings.database_url) as conn:
        has_key = load_gemini_key(conn, user_id, settings.master_key) is not None
    return {"has_key": has_key}


@router.put("/account/gemini-key", status_code=204)
def set_gemini_key(body: GeminiKeyBody, user_id: str = Depends(get_current_user_id),
                   settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        save_gemini_key(conn, user_id, settings.master_key, body.api_key)
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
