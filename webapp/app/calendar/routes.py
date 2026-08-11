"""Calendar suggestions: list what extraction has surfaced, then confirm
(create a real event on the user's own calendar) or dismiss.

Confirm-to-create by design - see `calendar_suggestions`' own schema.sql
comment: a commitment is the model's guess at what's owed and when, real
enough to surface as a suggestion but not to write to a user's calendar
unattended.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException

from ..config import Settings
from ..db import connect
from ..deps import get_current_user_id, get_settings
from ..tokens.store import DbTokenStore
from .client import CalendarClient, CalendarError

router = APIRouter(prefix="/calendar", tags=["calendar"])

_SELECT = """
    SELECT s.id, s.status, s.calendar_event_id, c.id, c.text, c.kind,
           c.due_phrase, c.due_at, c.owner, c.counterparty, c.message_id
    FROM calendar_suggestions s JOIN commitments c ON c.id = s.commitment_id
    WHERE s.user_id = %s
"""


def _row_to_dict(row) -> dict:
    (suggestion_id, status, calendar_event_id, commitment_id, text, kind,
     due_phrase, due_at, owner, counterparty, message_id) = row
    return {
        "id": str(suggestion_id), "status": status,
        "calendar_event_id": calendar_event_id, "commitment_id": str(commitment_id),
        "text": text, "kind": kind, "due_phrase": due_phrase,
        "due_at": due_at.isoformat() if due_at else None,
        "owner": owner, "counterparty": counterparty,
        # The commitment's source message - lets the UI offer "read the whole
        # email" the same way a chat citation does (GET /messages/{id}),
        # rather than leaving `text`'s one extracted sentence as the only
        # thing a person can see of where this came from.
        "message_id": message_id,
    }


@router.get("/suggestions")
def list_suggestions(status: str = "pending",
                     user_id: str = Depends(get_current_user_id),
                     settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        if status == "all":
            rows = conn.execute(_SELECT + " ORDER BY c.due_at", (user_id,)).fetchall()
        else:
            rows = conn.execute(
                _SELECT + " AND s.status = %s ORDER BY c.due_at",
                (user_id, status)).fetchall()
    return [_row_to_dict(r) for r in rows]


def _load_suggestion(conn, suggestion_id: str, user_id: str) -> dict:
    row = conn.execute(_SELECT + " AND s.id = %s", (user_id, suggestion_id)).fetchone()
    if row is None:
        raise HTTPException(404, "no such suggestion")
    return _row_to_dict(row)


@router.post("/suggestions/{suggestion_id}/confirm")
def confirm_suggestion(suggestion_id: str, user_id: str = Depends(get_current_user_id),
                       settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        suggestion = _load_suggestion(conn, suggestion_id, user_id)
        if suggestion["status"] != "pending":
            raise HTTPException(409, f"suggestion is already {suggestion['status']}")
        if not suggestion["due_at"]:
            raise HTTPException(400, "this commitment has no resolved due date")

        store = DbTokenStore(conn=conn, user_id=user_id,
                             master_key=settings.master_key).load()
        client = CalendarClient(settings.gmail_client_id, settings.gmail_client_secret,
                                tokens=store)
        description = (f"{suggestion['owner'] or 'someone'} -> "
                       f"{suggestion['counterparty'] or 'someone'}: {suggestion['text']}")
        try:
            event = client.create_event(
                suggestion["text"], description, date.fromisoformat(suggestion["due_at"]))
        except CalendarError as exc:
            raise HTTPException(502, str(exc)) from exc

        conn.execute(
            "UPDATE calendar_suggestions SET status = 'confirmed', "
            "calendar_event_id = %s, updated_at = now() WHERE id = %s",
            (event.get("id", ""), suggestion_id))
    return {"status": "confirmed", "calendar_event_id": event.get("id", "")}


@router.post("/suggestions/{suggestion_id}/dismiss")
def dismiss_suggestion(suggestion_id: str, user_id: str = Depends(get_current_user_id),
                       settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        suggestion = _load_suggestion(conn, suggestion_id, user_id)
        if suggestion["status"] != "pending":
            raise HTTPException(409, f"suggestion is already {suggestion['status']}")
        conn.execute(
            "UPDATE calendar_suggestions SET status = 'dismissed', updated_at = now() "
            "WHERE id = %s", (suggestion_id,))
    return {"status": "dismissed"}
