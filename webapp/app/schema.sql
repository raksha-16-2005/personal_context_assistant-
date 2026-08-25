-- Multi-tenant tables for the Gmail RAG web app. Same Postgres instance as
-- the ablation pgvector tables (index/store.py's chunks_<config> tables) -
-- a different namespace, not a different database, per the plan.
--
-- Every per-user table cascades on user deletion. That is deliberate, not a
-- convenience: the plan's /account endpoint has to actually delete a user's
-- data, and "delete the user row, the rest follows" is the only version of
-- that which cannot be forgotten one table at a time as the schema grows.

CREATE TABLE IF NOT EXISTS users (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    google_sub    text UNIQUE NOT NULL,   -- Google's stable per-account id ("sub" claim)
    email         text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- An IANA zone name (e.g. "Asia/Kolkata"), not a UTC offset - offsets
    -- silently go stale across DST transitions, zone names don't. Used to
    -- resolve "today"/"this week" in the router's date arms against the
    -- user's own day, not the server's.
    timezone      text NOT NULL DEFAULT 'UTC'
);

-- `CREATE TABLE IF NOT EXISTS` above only runs on a brand-new database - it
-- cannot add a column to a `users` table that already exists from before
-- this field did. This is what actually reaches an existing deployment.
ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone text NOT NULL DEFAULT 'UTC';

-- oauth_tokens/gemini_keys store ciphertext only (see app/security.py). The
-- column names say encrypted_* rather than just token_/key_ so a reviewer
-- reading a `SELECT *` output is told what they're looking at without having
-- to already know the contract.
CREATE TABLE IF NOT EXISTS oauth_tokens (
    user_id                 uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    encrypted_refresh_token bytea NOT NULL,
    encrypted_access_token  bytea NOT NULL,
    scope                   text NOT NULL,
    expires_at              timestamptz NOT NULL,
    issued_at               timestamptz NOT NULL DEFAULT now()
);

-- `encrypted_key_2` is an optional backup key: LLM.complete (llm/client.py)
-- tries every model under `encrypted_key` first, and only moves on to
-- `encrypted_key_2` once every model is quota-exhausted there - a second key
-- has its own separate free-tier quota, so this is real headroom, not just a
-- retry.
CREATE TABLE IF NOT EXISTS gemini_keys (
    user_id         uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    encrypted_key   bytea NOT NULL,
    encrypted_key_2 bytea,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE gemini_keys ADD COLUMN IF NOT EXISTS encrypted_key_2 bytea;

-- DB-backed twin of emailrag.corpus.gmail.SyncState - same three fields, one
-- row per user instead of one local JSON file.
CREATE TABLE IF NOT EXISTS sync_state (
    user_id             uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    history_id          text NOT NULL DEFAULT '',
    last_sync_utc       text NOT NULL DEFAULT '',
    messages_seen       integer NOT NULL DEFAULT 0,
    status              text NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'syncing', 'ready', 'error')),
    error_detail        text NOT NULL DEFAULT '',
    -- Set once the `backfill_history` job (app/ingestion/worker.py) finishes
    -- fetching everything before the fast first sync's recent-only window -
    -- what stops that one-time catch-up from re-running on every later
    -- incremental sync.
    full_history_synced boolean NOT NULL DEFAULT false
);

ALTER TABLE sync_state ADD COLUMN IF NOT EXISTS full_history_synced boolean NOT NULL DEFAULT false;

-- The durable copy of USER_INDEX_ROOT/<user_id>/ for a deployment with no
-- persistent disk - see app/ingestion/blob_store.py. One tar+gzipped row per
-- user, ~30-40 MB for a typical mailbox; a deployment with a real volume
-- never reads or writes this table at all (download is a no-op once local
-- disk already has the directory).
CREATE TABLE IF NOT EXISTS user_index_blobs (
    user_id     uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    data        bytea NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       text NOT NULL DEFAULT '',
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text NOT NULL,
    citations        jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, created_at);

-- `content` for a refused answer is the literal INSUFFICIENT_CONTEXT
-- sentinel (see emailrag.generation.synthesize's own docstring for why it's
-- a fixed string rather than prose) - exactly right for the CLI/eval, but
-- not something a chat UI should show a person verbatim. Persisting the
-- already-computed boolean alongside it lets the frontend render a natural
-- refusal for history exactly the same way it does for a live answer,
-- without re-deriving refusal by matching that string client-side.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS refused boolean NOT NULL DEFAULT false;

-- Same shape as emailrag.extraction.schema.Commitment, plus user_id. Kept as
-- its own table rather than reusing the eval harness's JSONL-file loader
-- (pipeline.py's `_load_commitments`) - this table is written by a
-- background worker and read by both the router's SQL arm and the calendar
-- suggestion flow, which wants a real row to reference by id.
CREATE TABLE IF NOT EXISTS commitments (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message_id        text NOT NULL,
    text              text NOT NULL,
    kind              text NOT NULL DEFAULT 'other',
    direction         text NOT NULL DEFAULT 'unclear',
    owner             text NOT NULL DEFAULT '',
    counterparty      text NOT NULL DEFAULT '',
    due_phrase        text NOT NULL DEFAULT '',
    due_at            date,
    due_precision     text NOT NULL DEFAULT '',
    due_ambiguous     boolean NOT NULL DEFAULT false,
    due_alternative   date,
    due_rolled_year   boolean NOT NULL DEFAULT false,
    confidence        real NOT NULL DEFAULT 0,
    model             text NOT NULL DEFAULT '',
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS commitments_user_due_idx ON commitments (user_id, due_at);

-- Extraction always rescans the same recent window rather than tracking
-- "already processed" state (see app/extraction/worker.py) - this is what
-- makes rescanning safe: re-inserting a commitment this job already stored
-- is a no-op, not a duplicate.
--
-- Deliberately *not* `(user_id, message_id, model, text)`, unlike the
-- single-user eval harness's own separate table
-- (extraction/schema.py's DDL) - that one really does want one row per
-- model, to compare a local model against a hosted one. This table has no
-- such comparison to make, and Gemini's own automatic quota-rotation
-- fallback (llm/client.py's GEMINI_FALLBACKS) means the *same* message can
-- get extracted under a *different* model name on a later rescan - with
-- `model` in the key, that silently produced two rows for one real
-- commitment instead of being caught by ON CONFLICT.
--
-- A prior version of this index did include `model`; dedup existing rows
-- before tightening it, since a UNIQUE index cannot be created over data
-- that already violates it. Safe to run on every startup - once deduped,
-- there is nothing left to delete.
DELETE FROM commitments a USING commitments b
WHERE a.ctid < b.ctid
  AND a.user_id = b.user_id AND a.message_id = b.message_id AND a.text = b.text;

DROP INDEX IF EXISTS commitments_unique_idx;
CREATE UNIQUE INDEX IF NOT EXISTS commitments_unique_idx
    ON commitments (user_id, message_id, text);

-- One suggestion per commitment. UNIQUE(commitment_id) is what makes
-- re-running extraction over the same message idempotent instead of piling
-- up duplicate suggestions for a commitment that was already reviewed.
CREATE TABLE IF NOT EXISTS calendar_suggestions (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    commitment_id      uuid NOT NULL UNIQUE REFERENCES commitments(id) ON DELETE CASCADE,
    user_id            uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status             text NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'confirmed', 'dismissed')),
    calendar_event_id  text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS calendar_suggestions_user_status_idx
    ON calendar_suggestions (user_id, status);

CREATE TABLE IF NOT EXISTS digest_settings (
    user_id        uuid PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    enabled        boolean NOT NULL DEFAULT false,
    send_hour_utc  smallint NOT NULL DEFAULT 13 CHECK (send_hour_utc BETWEEN 0 AND 23),
    last_sent_utc  timestamptz
);

-- One row per generated digest, in-app rather than emailed (see
-- app/digest/service.py's module docstring for why) - kept as a history,
-- not just "the latest", so GET /digest/latest is a plain query instead of
-- the only copy of state that exists.
CREATE TABLE IF NOT EXISTS digests (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS digests_user_idx ON digests (user_id, created_at DESC);

-- The ingestion/digest job queue. A DB-polled table, not Celery/Redis - see
-- the plan's "Revisit later": this is deliberately the simplest thing that
-- rate-limits concurrent work, not a permanent architecture decision.
CREATE TABLE IF NOT EXISTS jobs (
    id          bigserial PRIMARY KEY,
    type        text NOT NULL,
    user_id     uuid REFERENCES users(id) ON DELETE CASCADE,
    status      text NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts    integer NOT NULL DEFAULT 0,
    run_after   timestamptz NOT NULL DEFAULT now(),
    payload     jsonb NOT NULL DEFAULT '{}'::jsonb,
    last_error  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
-- What the worker polls: the oldest due, queued job. Partial index, since a
-- finished job never needs to be found this way again.
CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs (run_after)
    WHERE status = 'queued';
