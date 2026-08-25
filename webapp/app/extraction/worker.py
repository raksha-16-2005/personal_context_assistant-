"""Background commitment extraction: read a user's recent mail, run
`CommitmentExtractor` with *their own* pasted Gemini key - never a shared
key, same principle as /chat's per-user `LLM` - store new commitments, and
turn the ones with a resolved due date into calendar suggestions.

Enqueued after every sync (see jobs/runner.py's `_handle_sync`), and always
rescans the same recent window rather than tracking "already processed"
state - see `EXTRACTION_WINDOW_DAYS`. `commitments_unique_idx` (schema.sql)
is what makes that safe: reprocessing a message this job already saw is a
no-op insert, not a duplicate, so no separate bookkeeping is needed to make
repeated runs idempotent.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pyarrow.parquet as pq

from emailrag.extraction.extract import CommitmentExtractor
from emailrag.llm.client import LLM

from ..commitments import generate_calendar_suggestions, insert_commitments
from ..gemini_keys import load_gemini_keys
from ..ingestion.worker import messages_path

# Bounds extraction cost the same way corpus/gmail.py's own since_query()
# bounds the single-user flow - see that module's docstring. A user's own
# Gemini key pays for these calls, so "recent window, rescanned every sync"
# stays affordable without incremental tracking.
EXTRACTION_WINDOW_DAYS = 14


def extract_for_user(conn, settings, user_id: str) -> dict:
    """Extract commitments from `user_id`'s recent mail and generate
    calendar suggestions for the ones with a due date.

    A no-op, not a failure, if the user has never pasted a Gemini key or
    never synced - most users will do both before extraction has anything
    to work with, and there is nothing to charge a call to yet.
    """
    gemini_keys = load_gemini_keys(conn, user_id, settings.master_key)
    if not gemini_keys:
        return {"skipped": "no gemini key saved"}

    msg_path = messages_path(settings.user_index_root, user_id)
    if not msg_path.exists():
        return {"skipped": "no messages synced yet"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=EXTRACTION_WINDOW_DAYS)
    rows = pq.read_table(msg_path).to_pylist()
    recent = [r for r in rows if r.get("date_utc") and r["date_utc"] >= cutoff]

    extractor = CommitmentExtractor(llm=LLM(provider="gemini", api_key=gemini_keys))
    found = []
    for row in recent:
        found.extend(extractor.extract(row))

    new_count = insert_commitments(conn, user_id, found)
    suggested = generate_calendar_suggestions(conn, user_id)

    return {"messages_scanned": len(recent), "commitments_found": len(found),
            "commitments_new": new_count, "suggestions_created": suggested}
