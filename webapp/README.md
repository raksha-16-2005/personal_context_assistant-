# Multi-tenant Gmail RAG backend

Turns the single-user, local-only pipeline in `../src/emailrag` into a
multi-tenant web backend: Google login, per-user Gmail ingestion, a shared
retrieval/generation pipeline that answers each user's questions over their
own mailbox using their own pasted Gemini key, and persisted conversation
history. Built against the plan's phased design - see phases 0-3 below for
where this stands.

## What's built and tested

Every backend item below has a passing test against a **real** local
Postgres and, for the ingestion/pipeline/chat/extraction paths, a **real**
embedding model (`sentence-transformers/all-MiniLM-L6-v2`, already cached
locally) - not just mocks. `pytest -q` (from this directory, see below)
reproduces all of it: 87 tests. The frontend was driven end to end in a real
browser (Playwright, against the running server with a real Postgres) rather
than just unit-tested - see "Frontend" below for what that covered.

- **Auth** (`app/auth/`): Google OAuth as a Web application client
  (`gmail.readonly` + `calendar.events` + `openid email`), CSRF-safe via a
  signed `state` token, encrypted token storage (`app/tokens/store.py`,
  `app/security.py`), signed session cookies.
- **Ingestion** (`app/ingestion/worker.py`): `gmail.sync()` -> dedup/filter
  (reusing `corpus.filters.dedup_and_filter`, extracted from
  `scripts/build_corpus.py` so both corpora share one decision) -> thread
  reconstruction -> chunk -> embed -> BM25, written per user under
  `USER_INDEX_ROOT/<user_id>/`. Verified: a first sync builds a real,
  queryable index; a second sync merges rather than duplicates; the result
  opens with the **unmodified** `Pipeline` class and returns a correct
  answer to a real query.
- **Job queue** (`app/jobs/`): a DB-polled `jobs` table (`FOR UPDATE SKIP
  LOCKED`, so two workers never double-claim), a dispatcher
  (`runner.py`), and the login callback enqueueing a new user's first sync.
- **Chat** (`app/chat/routes.py`): `POST /chat` retrieves, reranks, and
  synthesizes a cited answer using *that user's own* pasted Gemini key
  (`LLM.api_key`, a small extension to `emailrag.llm.client`), backed by a
  `PipelinePool` that shares **one** loaded embedding model across every
  cached user (see "the RAM-multiplication fix" below) rather than one per
  user. Conversations and turns persist to Postgres; `GET /conversations` and
  `GET /conversations/{id}` resume them.
- **Commitment extraction** (`app/extraction/worker.py`): after every sync,
  an `extract_commitments` job rescans the last `EXTRACTION_WINDOW_DAYS` of
  a user's mail with `emailrag.extraction.CommitmentExtractor`, using *that
  user's own* pasted Gemini key. Always rescans the same window rather than
  tracking "already processed" state -
  `commitments_unique_idx` (`schema.sql`) is what makes that idempotent
  instead of a source of duplicates.
- **Calendar suggestions** (`app/calendar/`): every commitment with a
  resolved due date gets one pending suggestion (`app/commitments.py`).
  `GET /calendar/suggestions`, and confirm-to-create
  (`POST .../confirm`) calls a real Google Calendar API client
  (`app/calendar/client.py`) to create an all-day event on the user's own
  calendar - or `POST .../dismiss`, which never touches Google at all.
  Confirm-to-create by design: a commitment is a model's guess at what's
  owed and when, real enough to surface, not real enough to write to a
  user's calendar unattended.
- **Daily digest** (`app/digest/`): a scheduled `generate_digest` job
  (`schedule_due_digests`, polled from the same loop as the job queue)
  computes due-soon commitments, pending calendar suggestions, and new-mail
  count, and stores it - surfaced in-app via `GET /digest/latest`, not
  emailed. That's a scope decision: emailing it would need a new
  `gmail.send` scope and a re-consent round for every tester, for a feature
  an in-app panel already covers.
- **Account deletion** (`app/account/routes.py`): `DELETE /account` removes
  the user row (every other table cascades from it - tokens, keys,
  conversations, commitments, suggestions, digests, jobs), drops their
  cached `Pipeline` from the pool, deletes their
  `USER_INDEX_ROOT/<user_id>/` directory (the one thing Postgres cascade
  cannot reach), and clears the session cookie. `GET /me` and the
  `/account/gemini-key` routes back the frontend's own auth check and
  Settings page.
- **Frontend** (`frontend/`): a React SPA (Vite, `react-router-dom`) with
  Login, Chat, Commitments, Digest, and Settings pages, talking to the API
  above over plain `fetch()`. Built once (`npm run build`) and served
  *by* FastAPI itself in production (see "Frontend build" below) - there is
  no separate frontend deployment.

## What's deliberately not built yet

- **Google's app verification + security assessment** for going fully
  public - see "Phase 0" below. Nothing in this codebase can substitute for
  it - it's paperwork and process, not code.

## Phase 0 - steps only you can do

These aren't code, and nothing here can do them for you:

1. **Google Cloud project**: enable the Gmail API and the Calendar API.
2. **OAuth consent screen**: External, status **Testing**, add every early
   tester's email as a test user (Testing apps are capped at ~100). Add a
   Privacy Policy URL - Google requires one to even configure the consent
   screen, regardless of verification status.
3. **OAuth client**: Credentials -> Create -> OAuth client ID -> **Web
   application** (not Desktop - that's what `scripts/gmail_auth.py` uses
   locally, and it's a different client type with different redirect-URI
   rules). Add `<OAUTH_REDIRECT_BASE_URL>/auth/google/callback` as an
   authorized redirect URI.
4. Put the client id/secret in `webapp/.env` (copy `.env.example`).
5. **Know the Testing-status catch before you rely on this for real use**:
   refresh tokens for Gmail scopes expire after 7 days while the app is in
   Testing. Every tester will need to log in again weekly until the app is
   verified - the same clock `corpus/gmail.py`'s desktop flow has always
   been subject to, just now affecting every user instead of one.
6. **When ready to go past ~100 users or past 7-day logins**: Google's app
   verification, and - because `gmail.readonly` is a restricted scope - a
   security assessment (CASA). This is an external, sometimes-paid,
   multi-week process. Start the paperwork once the OAuth flow here is
   stable, not before - see the plan's Phase 0/7 for why submitting early
   just means resubmitting after scope changes.

## The RAM-multiplication fix, in case it needs re-explaining later

The naive design - one `Pipeline` per cached user - would load a full copy
of the embedding model per user, even though the weights are identical for
everyone (one shipped chunking/model config, `Settings.shipped_model`).
`PipelinePool` (`app/pipeline_pool.py`) loads the model **once**, and every
`Pipeline.__init__` call borrows it via the `model=` parameter added to
`emailrag/pipeline.py` for exactly this. `tests/test_pipeline_pool.py`
proves it with two *real* per-user indices: `pipe_a.model is pipe_b.model`.

## Frontend build

`frontend/` is a separate Vite project - `npm`, not `pip`, and its own
`node_modules`/`dist` (gitignored). FastAPI only serves the *build output*;
it never runs Vite itself.

```bash
cd webapp/frontend
npm install
npm run build              # writes frontend/dist - FastAPI serves this directly
```

For frontend-only iteration, `npm run dev` runs Vite's own dev server on
`:5173` and proxies API paths to `:8000` (see `vite.config.js`'s
`API_PREFIXES`) - run the backend (below) alongside it. There is no CORS
configuration anywhere because there is deliberately only ever one origin
in production: FastAPI serves both the API and the built app.

## Running it locally

```bash
cd webapp
cp .env.example .env      # fill in DATABASE_URL, MASTER_KEY, SESSION_SECRET,
                          # GMAIL_CLIENT_ID/SECRET (see Phase 0 above)
pip install -r requirements.txt      # same venv as ../requirements.txt is fine -
                                     # verified no conflict with the torch/
                                     # transformers pin on this project's own
                                     # Intel-macOS dev machine
uvicorn app.main:app --reload
```

Without a Google OAuth client (Phase 0 not done yet), the login button
redirects to Google and fails there - everything past login can still be
exercised by minting a session cookie directly:
`app.auth.session.create_session_cookie(session_secret, user_id)` for a user
row you've inserted yourself, then setting it as the `emailrag_session`
cookie. That's how this app's own frontend was verified end to end before
any real OAuth credentials existed.

The ingestion job queue (syncs, extraction, digests) runs as a background
thread inside `uvicorn app.main:app` itself by default - nothing extra to
run. Set `RUN_WORKER_IN_PROCESS=false` and run it as its own process instead
if you want a slow/stuck job isolated from `/chat` traffic:

```bash
python -m app.jobs.runner
```

## Tests

```bash
cd webapp
pytest -q                 # ~110s: several tests load the real embedding model
pytest -q -m "not slow"   # skip those, for a fast inner loop
```

Requires a local Postgres reachable at `postgresql:///emailrag` (the same
instance `../config.yaml`'s pgvector work already uses) - tests apply
`schema.sql` themselves and clean up the rows they create, but do not stand
up Postgres itself. These are backend-only; the frontend has no separate
test suite (it's small enough that the browser-driven check above stands
in for one) - `npm run build` failing is the fast signal something in
`frontend/src` broke.
