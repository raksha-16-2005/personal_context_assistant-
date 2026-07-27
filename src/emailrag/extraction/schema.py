"""The `commitments` table: what an extracted obligation is, and its DDL.

A commitment is the thing the router's SQL arm queries. "What's due this week" is
a `WHERE due_at BETWEEN ... ` over this table, not a similarity search - which is
the entire point of having a router, and the reason this table exists rather than
the system answering temporal questions by embedding them.

Three schema decisions worth defending:

**The deadline phrase is stored alongside the resolved date.** `due_at` is derived
from `due_phrase` plus the message's sent date (see dates.py), and a resolution
that turns out wrong is unfixable if the phrase is thrown away. Keeping both makes
the date-accuracy metric computable after the fact, and makes a convention change
("next Thursday" means the nearer one) a re-resolution rather than a re-extraction
at 25 s/message.

**Confidence and precision are separate columns.** A model can be certain the
deadline is "sometime next month" (high confidence, month precision) or unsure
whether a specific Thursday was meant (low confidence, day precision). Collapsing
them into one number loses the distinction that decides whether to show a date at
all.

**Every row names the model that produced it.** The extraction arm of this project
compares a local 3B model against a hosted one; rows from the two cannot be mixed
in a count without saying which is which, and a table that cannot answer "which
model claimed this" cannot support the comparison it exists for.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime

# Kinds of obligation. Kept small on purpose: a taxonomy with fifteen categories
# needs a second annotator to be credible, and the router only distinguishes
# "something is owed by a date" from "something is not".
KINDS = ("deliverable", "meeting", "payment", "review", "decision", "other")

DIRECTIONS = ("i_owe", "they_owe", "mutual", "unclear")


@dataclass(slots=True)
class Commitment:
    """One obligation extracted from one message."""

    message_id: str                  # dedup_key of the source message
    text: str                        # what is owed, in the extractor's words
    kind: str = "other"
    direction: str = "unclear"
    owner: str = ""                  # email address or name, verbatim
    counterparty: str = ""

    # The deadline, both as stated and as resolved. See the module docstring.
    due_phrase: str = ""
    due_at: date | None = None
    due_precision: str = ""
    due_ambiguous: bool = False
    due_alternative: date | None = None
    due_rolled_year: bool = False

    confidence: float = 0.0
    quote: str = ""                  # the sentence it came from, for audit
    model: str = ""
    extracted_utc: str = ""

    def as_row(self) -> dict:
        row = asdict(self)
        for key in ("due_at", "due_alternative"):
            value = row.get(key)
            row[key] = value.isoformat() if isinstance(value, (date, datetime)) else None
        return row


# Postgres DDL. Kept as a plain string rather than an ORM: this is one table, the
# project already talks to psycopg directly for pgvector, and a migration
# framework for a single CREATE TABLE is more moving parts than it removes.
DDL = """
CREATE TABLE IF NOT EXISTS commitments (
    id              BIGSERIAL PRIMARY KEY,
    message_id      TEXT        NOT NULL,
    text            TEXT        NOT NULL,
    kind            TEXT        NOT NULL DEFAULT 'other',
    direction       TEXT        NOT NULL DEFAULT 'unclear',
    owner           TEXT        NOT NULL DEFAULT '',
    counterparty    TEXT        NOT NULL DEFAULT '',

    -- Stored together deliberately: due_at is derived from due_phrase plus the
    -- message's sent date, and a wrong resolution is unfixable without the phrase.
    due_phrase      TEXT        NOT NULL DEFAULT '',
    due_at          DATE,
    due_precision   TEXT        NOT NULL DEFAULT '',
    due_ambiguous   BOOLEAN     NOT NULL DEFAULT FALSE,
    due_alternative DATE,
    due_rolled_year BOOLEAN     NOT NULL DEFAULT FALSE,

    confidence      REAL        NOT NULL DEFAULT 0,
    quote           TEXT        NOT NULL DEFAULT '',
    model           TEXT        NOT NULL DEFAULT '',
    extracted_utc   TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT commitments_kind_ck
        CHECK (kind IN ('deliverable','meeting','payment','review','decision','other')),
    CONSTRAINT commitments_direction_ck
        CHECK (direction IN ('i_owe','they_owe','mutual','unclear')),
    -- One model may only claim a given commitment once per message. Re-running
    -- extraction for the same arm updates rather than duplicating; a second model
    -- is a separate row on purpose, because the comparison needs both.
    CONSTRAINT commitments_unique UNIQUE (message_id, model, text)
);

-- The router's temporal arm is a range scan on due_at. Partial: rows with no
-- deadline are the majority and are never in a "what's due" answer.
CREATE INDEX IF NOT EXISTS commitments_due_at_idx
    ON commitments (due_at) WHERE due_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS commitments_message_idx ON commitments (message_id);
CREATE INDEX IF NOT EXISTS commitments_model_idx   ON commitments (model);
"""

DROP = "DROP TABLE IF EXISTS commitments;"

INSERT = """
INSERT INTO commitments (
    message_id, text, kind, direction, owner, counterparty,
    due_phrase, due_at, due_precision, due_ambiguous, due_alternative,
    due_rolled_year, confidence, quote, model
) VALUES (
    %(message_id)s, %(text)s, %(kind)s, %(direction)s, %(owner)s, %(counterparty)s,
    %(due_phrase)s, %(due_at)s, %(due_precision)s, %(due_ambiguous)s,
    %(due_alternative)s, %(due_rolled_year)s, %(confidence)s, %(quote)s, %(model)s
)
ON CONFLICT (message_id, model, text) DO UPDATE SET
    due_at = EXCLUDED.due_at,
    due_phrase = EXCLUDED.due_phrase,
    due_precision = EXCLUDED.due_precision,
    due_ambiguous = EXCLUDED.due_ambiguous,
    due_alternative = EXCLUDED.due_alternative,
    due_rolled_year = EXCLUDED.due_rolled_year,
    confidence = EXCLUDED.confidence,
    extracted_utc = now();
"""


def validate(c: Commitment) -> list[str]:
    """Problems that would make a row misleading rather than merely imperfect."""
    problems = []
    if not c.message_id:
        problems.append("no message_id - the row cannot be traced to a source")
    if not c.text.strip():
        problems.append("empty text")
    if c.kind not in KINDS:
        problems.append(f"unknown kind {c.kind!r} (expected one of {list(KINDS)})")
    if c.direction not in DIRECTIONS:
        problems.append(f"unknown direction {c.direction!r}")
    if not 0.0 <= c.confidence <= 1.0:
        problems.append(f"confidence {c.confidence} outside [0, 1]")
    # A resolved date with no phrase cannot be re-resolved or audited, which is
    # the one thing the schema exists to guarantee.
    if c.due_at is not None and not c.due_phrase:
        problems.append("due_at set with no due_phrase - not auditable")
    return problems
