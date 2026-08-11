"""Web-app OAuth flow, built on corpus/gmail.py's token-exchange primitives.

The token exchange itself (`exchange_code`, `refresh_access_token`) is plain
HTTP with no notion of "desktop" or "web" baked in, so it's reused unchanged.
Two things the desktop flow never needed are added here instead of there:
extra scopes (Calendar, plus `openid email` for a stable account id), and a
userinfo call to actually get that id - the desktop flow only ever needed a
mailbox to read, never an account to distinguish from other accounts.
"""
from __future__ import annotations

import json
import urllib.request

from emailrag.corpus import gmail as G

USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Gmail read + Calendar write (for the confirm-to-create suggestion flow) +
# openid/email (for a stable per-account id and an address to show in the
# UI - the desktop flow shows a mailbox, never distinguishes accounts).
WEB_SCOPES = f"{G.SCOPE} {G.CALENDAR_SCOPE} openid email"

TIMEOUT = 30


def login_url(client_id: str, redirect_uri: str, state: str) -> str:
    return G.authorization_url(client_id, redirect_uri, scopes=WEB_SCOPES, state=state)


def fetch_userinfo(access_token: str) -> dict:
    """`sub` (stable account id) and `email`, given a fresh access token."""
    req = urllib.request.Request(
        USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())
