"""Google Calendar API v3, just the one call the confirm-to-create flow
needs: creating an event on the user's primary calendar.

Plain urllib, matching every other Google API call in this project
(corpus/gmail.py) rather than pulling in google-api-python-client - same
reasoning as that module's own docstring. Reuses `ensure_access_token` from
there too: one OAuth grant covers both the Gmail and Calendar scopes, so
refreshing here would just be a second, driftable copy of the same logic.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, timedelta

from emailrag.corpus.gmail import TokenStoreBase, ensure_access_token

API = "https://www.googleapis.com/calendar/v3"
TIMEOUT = 30


class CalendarError(RuntimeError):
    pass


class CalendarClient:
    def __init__(self, client_id: str, client_secret: str, tokens: TokenStoreBase) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.tokens = tokens

    def create_event(self, summary: str, description: str, due_date: date) -> dict:
        """An all-day event on `due_date`.

        Google's API treats `end.date` as exclusive, so a one-day event needs
        the day *after* as its end - worth stating explicitly, since getting
        it backwards silently creates a zero-length event that never appears
        on the day it's meant to.
        """
        body = {
            "summary": summary[:1000],
            "description": description[:5000],
            "start": {"date": due_date.isoformat()},
            "end": {"date": (due_date + timedelta(days=1)).isoformat()},
        }
        return self._post("/calendars/primary/events", body)

    def _post(self, path: str, body: dict) -> dict:
        token = ensure_access_token(self.client_id, self.client_secret, self.tokens)
        req = urllib.request.Request(
            f"{API}{path}", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:400]
            raise CalendarError(f"calendar API HTTP {exc.code}: {detail}") from exc
