"""Shared FastAPI dependencies: settings (loaded once), per-request auth, and
the app-wide PipelinePool (one shared embedding model, see pipeline_pool.py).
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Cookie, HTTPException, Request, status

from .auth.session import COOKIE_NAME, read_session_cookie
from .config import Settings, load_settings


@lru_cache
def get_settings() -> Settings:
    return load_settings()


def get_pipeline_pool(request: Request):
    """The PipelinePool built once in main.py's lifespan and hung off
    app.state - not its own lru_cache'd factory, since it needs settings
    that are already resolved by the time the app starts, and one process
    must share exactly one pool, never a second one built by a stray call
    before startup."""
    return request.app.state.pipeline_pool


def get_current_user_id(
    emailrag_session: str | None = Cookie(default=None),
) -> str:
    settings = get_settings()
    if not emailrag_session:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not logged in")
    user_id = read_session_cookie(settings.session_secret, emailrag_session)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session expired or invalid")
    return user_id
