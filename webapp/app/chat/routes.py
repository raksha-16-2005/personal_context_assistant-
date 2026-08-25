"""POST /chat: retrieve, rerank, synthesize a cited answer using the caller's
own pasted Gemini key, and persist the turn. GET endpoints resume history.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from emailrag.generation.synthesize import Synthesizer
from emailrag.llm.client import LLM

from ..commitments import load_commitments_for_router
from ..config import Settings
from ..db import connect
from ..deps import get_current_user_id, get_pipeline_pool, get_settings
from ..gemini_keys import load_gemini_key
from ..pipeline_pool import PipelinePool
from ..users import get_email, get_timezone

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict]
    refused: bool


def _require_conversation(conn, conversation_id: str, user_id: str) -> None:
    owned = conn.execute(
        "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
        (conversation_id, user_id)).fetchone()
    if owned is None:
        raise HTTPException(404, "no such conversation")


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user_id: str = Depends(get_current_user_id),
        settings: Settings = Depends(get_settings),
        pool: PipelinePool = Depends(get_pipeline_pool)):
    with connect(settings.database_url) as conn:
        status_row = conn.execute(
            "SELECT status FROM sync_state WHERE user_id = %s", (user_id,)).fetchone()
        if status_row is None or status_row[0] != "ready":
            raise HTTPException(
                409, "your mailbox is still syncing - try again shortly")

        gemini_key = load_gemini_key(conn, user_id, settings.master_key)
        if not gemini_key:
            raise HTTPException(400, "paste your Gemini API key in settings first")

        conversation_id = body.conversation_id
        if conversation_id is None:
            row = conn.execute(
                "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                (user_id, body.question[:80])).fetchone()
            conversation_id = str(row[0])
        else:
            _require_conversation(conn, conversation_id, user_id)

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) "
            "VALUES (%s, 'user', %s)",
            (conversation_id, body.question))

        pipeline = pool.get(user_id, conn=conn)
        # One LLM, this user's own, shared by both consumers that need one -
        # swapped in per request rather than baked into the cached Pipeline,
        # since this user's key may have changed since their last question.
        # Without this, the router's rules-abstained fallback would build its
        # *own* keyless `LLM()` (see QueryRouter.llm's property), which in
        # this multi-tenant process has no env-var key to find at all - it
        # would silently fail every classification instead of ever really
        # asking a model, rather than fail loudly.
        llm = LLM(provider="gemini", api_key=gemini_key)
        pipeline.synthesizer = Synthesizer(llm=llm)
        if pipeline.router is not None:
            pipeline.router._llm = llm
        # Same reasoning, same pattern: extraction runs in a separate process
        # (the job runner) that this cached Pipeline has no way to be
        # notified by, so the router's commitments are re-read fresh here
        # rather than trusted from whenever this Pipeline was first built.
        pipeline.commitments = load_commitments_for_router(conn, user_id)
        # `Pipeline.search`'s own `as_of` default is the *corpus's* latest
        # message date - deliberate for the offline eval harness, where
        # reproducibility matters more than wall-clock time, but wrong here:
        # a live chat service's "today" is the real date, not whenever this
        # user's last-synced email happened to arrive. Getting this wrong is
        # exactly what silently anchored "this week" on the wrong week.
        #
        # And "the real date" is *this user's* real date, not the server's -
        # resolved in UTC alone, "this week" flips to the wrong week for
        # anyone east of UTC during the hours after their local midnight but
        # before UTC's. `set_timezone` already rejects an unresolvable zone
        # name before it ever reaches the users table, so this only fails
        # here if that guarantee were ever bypassed - fall back to UTC rather
        # than fail the question over a timezone problem.
        try:
            tz = ZoneInfo(get_timezone(conn, user_id))
        except ZoneInfoNotFoundError:
            tz = timezone.utc
        today = datetime.now(tz).date().isoformat()
        # `tz` here too, not just `as_of` - `_message_date_arm` converts each
        # message's UTC timestamp into this same zone before comparing, so
        # the window and the messages being checked against it agree about
        # what day "today" was. Passing only `as_of` would resolve the
        # window in the user's local day while still bucketing messages by
        # their UTC day - the two would disagree for part of every day.
        # Without this, nothing in the prompt ever says whose mailbox is being
        # searched - see Synthesizer.answer's own docstring for why "how many
        # emails did I get today" is otherwise unanswerable in principle, not
        # just a retrieval miss.
        answer, _result = pipeline.ask(body.question, as_of=today, tz=tz,
                                       mailbox_owner=get_email(conn, user_id),
                                       include_route_notes=True)

        # Only the sources the answer actually cites, not every source it was
        # offered - `answer.citations` includes up to `n_sources` distractors
        # the model never referenced, and showing those in a UI reads as the
        # same handful of senders/subjects repeated for no reason. See
        # `Answer.cited_sources`'s own docstring in generation/synthesize.py.
        citations = [asdict(c) for c in answer.cited_sources]
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, citations, refused) "
            "VALUES (%s, 'assistant', %s, %s, %s)",
            (conversation_id, answer.text, json.dumps(citations), answer.refused))

    return ChatResponse(conversation_id=conversation_id, answer=answer.text,
                        citations=citations, refused=answer.refused)


@router.get("/messages/{message_id}")
def get_message(message_id: str, user_id: str = Depends(get_current_user_id),
                settings: Settings = Depends(get_settings)):
    """The full email behind a citation - sender/recipients/cc/date/subject and
    the whole body, not the ~1400-char clip `format_sources` truncates the
    prompt to. Reads straight from this user's own `messages.parquet`
    (ingestion/worker.py), which is what naturally scopes this to their own
    mailbox: there is no other user's messages this path could ever open.
    """
    import pyarrow.parquet as pq

    from ..ingestion.worker import messages_path

    path = messages_path(settings.user_index_root, user_id)
    if not path.exists():
        raise HTTPException(404, "message not found")

    table = pq.read_table(
        path, columns=["dedup_key", "sender", "recipients", "cc", "subject",
                       "date_utc", "body"],
        filters=[("dedup_key", "=", message_id)])
    if table.num_rows == 0:
        raise HTTPException(404, "message not found")

    row = table.to_pylist()[0]
    date = row.get("date_utc")
    return {
        "message_id": message_id,
        "sender": row.get("sender") or "",
        "recipients": row.get("recipients") or "",
        "cc": row.get("cc") or "",
        "subject": row.get("subject") or "",
        "date": date.isoformat() if date is not None else "",
        "body": row.get("body") or "",
    }


@router.get("/conversations")
def list_conversations(user_id: str = Depends(get_current_user_id),
                       settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations WHERE user_id = %s "
            "ORDER BY created_at DESC", (user_id,)).fetchall()
    return [{"id": str(r[0]), "title": r[1], "created_at": r[2].isoformat()} for r in rows]


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = Depends(get_current_user_id),
                     settings: Settings = Depends(get_settings)):
    with connect(settings.database_url) as conn:
        _require_conversation(conn, conversation_id, user_id)
        rows = conn.execute(
            "SELECT role, content, citations, refused, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY created_at", (conversation_id,)).fetchall()
    return [{"role": r[0], "content": r[1], "citations": r[2], "refused": r[3],
            "created_at": r[4].isoformat()}
           for r in rows]
