"""Persisting extracted commitments, and turning the ones with a resolved
due date into calendar suggestions.

`insert_commitments` is idempotent by design: `commitments_unique_idx` (see
schema.sql) means re-extracting the same recent window on every sync
(`app/extraction/worker.py` deliberately rescans rather than tracking
"already processed" state) never duplicates a row it already has - even
across a rescan that happened to land on a different Gemini fallback model
than the one before it (see that index's own comment for why `model` is
deliberately not part of the conflict target here).
"""
from __future__ import annotations

from emailrag.extraction.schema import Commitment

_INSERT = """
INSERT INTO commitments (
    user_id, message_id, text, kind, direction, owner, counterparty,
    due_phrase, due_at, due_precision, due_ambiguous, due_alternative,
    due_rolled_year, confidence, model
) VALUES (
    %(user_id)s, %(message_id)s, %(text)s, %(kind)s, %(direction)s,
    %(owner)s, %(counterparty)s, %(due_phrase)s, %(due_at)s,
    %(due_precision)s, %(due_ambiguous)s, %(due_alternative)s,
    %(due_rolled_year)s, %(confidence)s, %(model)s
)
ON CONFLICT (user_id, message_id, text) DO NOTHING
RETURNING id
"""


def insert_commitments(conn, user_id: str, commitments: list[Commitment]) -> int:
    """Insert new commitments for `user_id`, skipping ones already stored.

    Returns how many were actually new - the caller uses this to decide
    whether generating calendar suggestions is worth a query at all.
    """
    inserted = 0
    for c in commitments:
        row = conn.execute(_INSERT, {
            "user_id": user_id, "message_id": c.message_id, "text": c.text,
            "kind": c.kind, "direction": c.direction, "owner": c.owner,
            "counterparty": c.counterparty, "due_phrase": c.due_phrase,
            "due_at": c.due_at, "due_precision": c.due_precision,
            "due_ambiguous": c.due_ambiguous, "due_alternative": c.due_alternative,
            "due_rolled_year": c.due_rolled_year, "confidence": c.confidence,
            "model": c.model,
        }).fetchone()
        if row is not None:
            inserted += 1
    return inserted


def generate_calendar_suggestions(conn, user_id: str) -> int:
    """One pending suggestion per commitment that has a due date and doesn't
    already have one. `calendar_suggestions.commitment_id` is UNIQUE, so
    this is safe to call after every extraction pass, not just the first.
    """
    rows = conn.execute(
        """
        INSERT INTO calendar_suggestions (commitment_id, user_id)
        SELECT c.id, c.user_id FROM commitments c
        WHERE c.user_id = %s AND c.due_at IS NOT NULL
        ON CONFLICT (commitment_id) DO NOTHING
        RETURNING id
        """,
        (user_id,),
    ).fetchall()
    return len(rows)


_SELECT_FOR_ROUTER = """
    SELECT message_id, text, kind, direction, owner, counterparty, due_phrase,
           due_at, due_precision, due_ambiguous, due_alternative,
           due_rolled_year, confidence, model
    FROM commitments WHERE user_id = %s
"""


def load_commitments_for_router(conn, user_id: str) -> list[Commitment]:
    """This user's commitments as `Commitment` rows, for `Pipeline`'s router.

    Read fresh on every `/chat` call (see chat/routes.py) rather than baked
    into the cached `Pipeline` at construction time - `PipelinePool` caches
    Pipelines across requests, but extraction runs in the separate job-runner
    process (see app/jobs/runner.py), so a Pipeline built before this user's
    most recent extraction pass would otherwise never see rows it added.
    Cheap enough (a handful of rows per user) that re-querying every request
    is not worth caching around.
    """
    rows = conn.execute(_SELECT_FOR_ROUTER, (user_id,)).fetchall()
    return [
        Commitment(
            message_id=r[0], text=r[1], kind=r[2], direction=r[3], owner=r[4],
            counterparty=r[5], due_phrase=r[6], due_at=r[7], due_precision=r[8],
            due_ambiguous=r[9], due_alternative=r[10], due_rolled_year=r[11],
            confidence=r[12], model=r[13],
        )
        for r in rows
    ]
