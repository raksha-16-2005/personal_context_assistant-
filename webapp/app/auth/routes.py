"""Login with Google, exchange the code, upsert the user, store their
encrypted tokens, issue a session cookie, and enqueue their first sync.
"""
from __future__ import annotations

import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from emailrag.corpus import gmail as G

from ..config import Settings
from ..db import connect
from ..deps import get_settings
from ..jobs.queue import enqueue
from ..tokens.store import DbTokenStore
from ..users import upsert_user
from .google_oauth import fetch_userinfo, login_url
from .session import (COOKIE_NAME, SESSION_MAX_AGE_SECONDS, create_session_cookie,
                      create_state_token, verify_state_token)

router = APIRouter(prefix="/auth/google", tags=["auth"])


@router.get("/login")
def login(settings: Settings = Depends(get_settings)):
    state = create_state_token(settings.session_secret, secrets.token_urlsafe(16))
    url = login_url(settings.gmail_client_id, settings.oauth_redirect_uri, state)
    return RedirectResponse(url)


@router.get("/callback")
def callback(code: str = "", state: str = "", error: str = "",
            settings: Settings = Depends(get_settings)):
    if error:
        raise HTTPException(400, f"Google denied access: {error}")
    if not code:
        raise HTTPException(400, "no authorization code returned")
    if not verify_state_token(settings.session_secret, state):
        raise HTTPException(
            400, "invalid or expired OAuth state - please try logging in again")

    tokens = G.exchange_code(settings.gmail_client_id, settings.gmail_client_secret,
                             code, settings.oauth_redirect_uri)
    if not tokens.get("refresh_token"):
        raise HTTPException(
            400,
            "Google returned no refresh token. This happens when the app was "
            "already authorised without a fresh consent prompt; revoke access "
            "at https://myaccount.google.com/permissions and try again.")

    userinfo = fetch_userinfo(tokens["access_token"])
    google_sub = userinfo.get("sub", "")
    email = userinfo.get("email", "")
    if not google_sub:
        raise HTTPException(400, "Google did not return an account id")

    with connect(settings.database_url) as conn:
        user_id = upsert_user(conn, google_sub, email)

        store = DbTokenStore(conn=conn, user_id=user_id, master_key=settings.master_key)
        store.data = {
            "refresh_token": tokens["refresh_token"],
            "access_token": tokens.get("access_token", ""),
            "expires_at": time.time() + int(tokens.get("expires_in", 3600)),
            "issued_at": time.time(),
            "scope": tokens.get("scope", ""),
        }
        store.save()

        # Only a brand-new account needs a full sync enqueued here - a
        # returning user's mailbox is kept current by incremental sync jobs
        # instead (see app/ingestion), which read sync_state.history_id
        # rather than starting over.
        existing = conn.execute(
            "SELECT 1 FROM sync_state WHERE user_id = %s", (user_id,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO sync_state (user_id, status) VALUES (%s, 'pending')",
                (user_id,))
            enqueue(conn, "initial_sync", user_id=user_id)

    response = RedirectResponse(url=settings.frontend_base_url)
    # `secure` follows the redirect base URL rather than a separate flag: a
    # cookie marked secure is silently dropped by the browser over plain
    # HTTP, which would otherwise make local dev (http://localhost) log
    # everyone straight back out after "logging in". A cross-site frontend
    # (settings.cookie_samesite="none") requires secure regardless.
    secure = settings.oauth_redirect_base_url.startswith("https://")
    response.set_cookie(
        COOKIE_NAME, create_session_cookie(settings.session_secret, user_id),
        httponly=True, secure=secure, samesite=settings.cookie_samesite,
        max_age=SESSION_MAX_AGE_SECONDS,
    )
    return response


@router.post("/logout")
def logout(settings: Settings = Depends(get_settings)):
    response = RedirectResponse(url=settings.frontend_base_url, status_code=303)
    secure = settings.oauth_redirect_base_url.startswith("https://")
    response.delete_cookie(
        COOKIE_NAME, secure=secure, samesite=settings.cookie_samesite)
    return response
