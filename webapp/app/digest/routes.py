"""GET/PUT digest settings, and GET the most recently generated digest.

See service.py's module docstring for why this is in-app rather than
emailed.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..config import Settings
from ..db import connect
from ..deps import get_current_user_id, get_settings
from . import service

router = APIRouter(prefix="/digest", tags=["digest"])


class DigestSettingsBody(BaseModel):
    enabled: bool
    send_hour_utc: int = Field(default=13, ge=0, le=23)


@router.get("/settings")
def get_digest_settings(user_id: str = Depends(get_current_user_id),
                        settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        return service.get_settings_row(conn, user_id)


@router.put("/settings")
def update_digest_settings(body: DigestSettingsBody,
                           user_id: str = Depends(get_current_user_id),
                           settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        service.save_settings(conn, user_id, body.enabled, body.send_hour_utc)
        return service.get_settings_row(conn, user_id)


@router.get("/latest")
def latest_digest(user_id: str = Depends(get_current_user_id),
                  settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        return service.get_latest(conn, user_id) or {}
