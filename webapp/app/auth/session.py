"""Signed cookies for the session and the OAuth CSRF `state` value.

Signed, not encrypted - neither carries a secret (a user id, or a random
nonce), only something that must not be forgeable without SESSION_SECRET.
`itsdangerous` is what Flask's own session cookie is built on; reused here
rather than hand-rolling HMAC-over-JSON.
"""
from __future__ import annotations

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

COOKIE_NAME = "emailrag_session"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 3600     # 30 days

# Short-lived: a state token is only meant to survive the round trip to
# Google's consent screen and back, not to be replayable minutes later.
STATE_MAX_AGE_SECONDS = 600


def _serializer(secret: str, salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt=salt)


def create_session_cookie(secret: str, user_id: str) -> str:
    return _serializer(secret, "session").dumps({"user_id": user_id})


def read_session_cookie(secret: str, cookie_value: str) -> str | None:
    """The user_id the cookie names, or None if it's missing, forged, or expired."""
    try:
        data = _serializer(secret, "session").loads(
            cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


def create_state_token(secret: str, nonce: str) -> str:
    """A signed, short-lived token to pass as OAuth's `state` param.

    Stateless on purpose - no server-side storage to look it up in, so a
    horizontally-scaled backend (or just a restarted one) can still verify a
    state token issued moments ago by a different process.
    """
    return _serializer(secret, "oauth-state").dumps({"nonce": nonce})


def verify_state_token(secret: str, token: str) -> bool:
    try:
        _serializer(secret, "oauth-state").loads(token, max_age=STATE_MAX_AGE_SECONDS)
        return True
    except (BadSignature, SignatureExpired):
        return False
