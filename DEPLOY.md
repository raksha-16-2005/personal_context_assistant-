# Deploying

Two independent paths - pick one, they don't mix:

- **Render + Neon** (below): genuinely free, no credit card anywhere. The
  backend has no persistent disk, which this app's code already accounts
  for - see "How this works on a disk-less free tier" below for what that
  means in practice.
- **Fly.io** (further down): simpler - one platform, a real persistent
  volume, no rehydration-from-Postgres behavior to think about - but
  requires a card on file and runs roughly $5-10/mo for what this app
  actually uses.

Both serve the frontend from the same origin as the backend by default -
one URL, works everywhere, no further setup. Splitting the frontend onto
Vercel is possible but **not recommended**; see its own section below for
why (a real, hit-in-practice login bug in Safari).

## Deploying to Render + Neon (free tier)

### 0. Neon: create the free Postgres database

1. Sign up at neon.tech (no card).
2. Create a project, then copy its connection string (Dashboard -> Connection
   Details) - it looks like
   `postgresql://<user>:<password>@<host>/<db>?sslmode=require`. This is
   `DATABASE_URL`.

512 MB storage on the free plan - comfortably enough for a handful of users
at this app's own measured ~30-40 MB/user (see the storage note below), not
meant for a real multi-tenant deployment at any scale.

### 1. Render: deploy the backend

1. render.com -> New -> Web Service -> connect this GitHub repo.
2. **Runtime**: Docker (Render detects the root `Dockerfile` automatically).
3. **Root Directory**: leave blank (repo root - the Dockerfile's `COPY`
   paths assume it).
4. **Instance Type**: Free.
5. **Health Check Path**: `/healthz`.
6. **Environment Variables**:
   ```
   DATABASE_URL=<Neon connection string from step 0>
   MASTER_KEY=<python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'>
   SESSION_SECRET=<python3 -c 'import secrets; print(secrets.token_urlsafe(32))'>
   GMAIL_CLIENT_ID=<from Google Cloud Console>
   GMAIL_CLIENT_SECRET=<from Google Cloud Console>
   OAUTH_REDIRECT_BASE_URL=https://<your-service>.onrender.com
   ```
   (`<your-service>` is whatever you name the service - Render shows you the
   `.onrender.com` URL before you finish creating it.)
7. Create the service and let the first deploy finish.

`USER_INDEX_ROOT` and `RUN_WORKER_IN_PROCESS` need no entry here - their
defaults already do the right thing on Render (see the disk-less note
below).

### 2. Google OAuth client - redirect URI

Google Cloud Console -> APIs & Services -> Credentials -> your OAuth client
-> Authorized redirect URIs, add:

```
https://<your-service>.onrender.com/auth/google/callback
```

### 3. Check it

Nothing more to deploy - this backend serves its own built frontend at `/`
(the Dockerfile builds it in). Open
`https://<your-service>.onrender.com`, sign in with Google, and confirm you
land back on it logged in. Leave `FRONTEND_BASE_URL` and
`SESSION_COOKIE_SAMESITE` unset - their defaults are exactly right for this
single-origin setup, and setting them (e.g. for the Vercel split below)
would break login here instead.

### How this works on a disk-less free tier

Render's free web services have no persistent disk - everything written
locally is wiped on every redeploy and every wake from its 15-minute idle
sleep. This app's per-user mailbox index (`USER_INDEX_ROOT`) is exactly that
kind of local state, so it's backed by a `user_index_blobs` table in
Postgres (`webapp/app/ingestion/blob_store.py`): every sync saves a fresh
copy there, and it's restored to local disk automatically the next time
it's needed, if local disk doesn't already have it. Nothing to configure -
it's automatic - but it does mean:

- **The first request after a 15-minute-idle wake is slow** - it has to
  rehydrate a mailbox's index from Postgres (~30-40 MB for a typical user)
  before it can answer, not just cold-start the process.
- **Neon's 512 MB free tier caps how many users this comfortably fits** -
  fine for personal use or a few testers, not a real multi-tenant rollout.

The background job worker (syncs, extraction, digests) also needed a home -
Render's free tier has no separate background-worker service type, so it
runs as a thread inside the same web service process instead
(`RUN_WORKER_IN_PROCESS=true`, the default - see `webapp/app/jobs/runner.py`
and `webapp/app/main.py`'s lifespan).

---

## Deploying to Fly.io (paid, simpler)

One-time setup, then a normal `fly deploy` for every update after. Run
these from the repo root (same directory as `Dockerfile` and `fly.toml`).

### 0. Install flyctl and log in

```bash
brew install flyctl
fly auth login          # opens a browser - this step needs you
```

### 1. Create the app (picks your unique name)

```bash
fly apps create           # interactive - suggests a name, or type your own
```

Whatever name it gives you, put it in **two** places:
- `fly.toml`'s `app = "..."` line
- `fly.toml`'s `OAUTH_REDIRECT_BASE_URL = "https://<name>.fly.dev"` line

### 2. Postgres

```bash
fly postgres create --name <name>-db     # pick the free/hobby tier when asked
fly postgres attach <name>-db --app <name>
```

`attach` sets `DATABASE_URL` as a secret automatically - you don't set it
by hand.

### 3. Persistent volume (for USER_INDEX_ROOT - per-user mailbox indices)

```bash
fly volumes create emailrag_data --app <name> --region iad --size 3
```

Match `--region` to `fly.toml`'s `primary_region`.

### 4. Secrets (never in fly.toml - these are the ones that matter if leaked)

```bash
fly secrets set --app <name> \
  MASTER_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  GMAIL_CLIENT_ID="<from Google Cloud Console>" \
  GMAIL_CLIENT_SECRET="<from Google Cloud Console>"
```

### 5. Google OAuth client - redirect URI must match the real URL

Back in Google Cloud Console (APIs & Services -> Credentials -> your OAuth
client -> Authorized redirect URIs), add:

```
https://<name>.fly.dev/auth/google/callback
```

(`localhost:8000` can stay too, if you still want local testing to work.)

### 6. Deploy

```bash
fly deploy
```

This runs `release_command` (`python -m app.security` - refuses to roll out
if any stored secret isn't real ciphertext, see that module's docstring)
before traffic ever reaches the new version, then starts the `app` process
group defined in `fly.toml`. The job runner (syncs, extraction, digests)
runs as a background thread inside that same process - no separate process
group or `fly scale count` step needed (a persistent volume means it never
needs the Postgres-rehydration path either - local disk is just always
there).

### 7. Check it

```bash
fly status --app <name>
curl https://<name>.fly.dev/healthz
open https://<name>.fly.dev
```

### Updating after the first deploy

Just `fly deploy` again - it rebuilds the image, runs the release command,
and does a rolling replace of the running machines.

---

## Splitting the frontend onto Vercel

**Not recommended - read this first.** Splitting the frontend onto a
different domain than the backend makes the session cookie a cross-site
cookie, and Safari's "Prevent Cross-Site Tracking" (on by default) silently
refuses to send it back to the backend: Google login appears to succeed,
then bounces straight back to the login page, logged out, with no error
shown anywhere - hit in practice deploying exactly this setup. Chrome
currently tolerates it. There is no code-level fix for this short of
putting the frontend and backend on subdomains of one domain you own (so
they're same-site, not just both HTTPS) - if you don't have a domain for
that, the single-origin setup above (no Vercel) is the one that actually
works in every browser, not just some.

If you still want this - a separate domain/CI for the frontend is worth
more to you than Safari support, say - here's how:

1. **Deploy the backend first** (Render or Fly, above) and note its public
   URL.

2. **Import the repo into Vercel** (vercel.com -> Add New -> Project), and
   set:
   - **Root Directory**: `webapp/frontend`
   - **Framework Preset**: Vite (auto-detected)
   - **Environment Variable**: `VITE_API_BASE_URL` = your backend's URL

   `webapp/frontend/vercel.json` already rewrites every path to
   `index.html`, so a hard refresh on a client-side route (`/chat`,
   `/commitments`, ...) doesn't 404.

3. **Deploy on Vercel** and note the URL it gives you,
   `https://<project>.vercel.app`.

4. **Point the backend back at it** - the frontend and backend are now on
   different origins, so the backend needs to know the frontend's URL (for
   CORS and for where to redirect after login/logout) and needs the
   session cookie to survive a cross-site request. On Render, set these as
   environment variables (Settings -> Environment); on Fly:

   ```bash
   fly deploy --app <name> \
     --env FRONTEND_BASE_URL=https://<project>.vercel.app \
     --env SESSION_COOKIE_SAMESITE=none
   ```

   (Or edit `fly.toml`'s `[env]` block directly - the two lines commented
   out there - and run `fly deploy` normally; either way this requires a
   redeploy, `fly secrets set` alone won't pick up `[env]` changes.)

5. **Add a second Google OAuth redirect URI.** The redirect URI is still
   the *backend's* `/auth/google/callback` - Vercel's URL never needs to be
   registered with Google.

6. **Check it**: open `https://<project>.vercel.app`, sign in with Google,
   and confirm you land back on the Vercel app logged in (not on the
   backend's own URL) - that confirms `FRONTEND_BASE_URL` and the cookie
   are both wired correctly.

## If something won't start

**Render**: check the service's Logs tab in the dashboard.

**Fly**:
```bash
fly logs --app <name>
```

The most likely first-deploy failures on either platform: a secret typo'd
or missing (`load_settings()` raises `ConfigError` naming the exact key), or
the OAuth redirect URI not matching exactly (Google returns
`redirect_uri_mismatch` and nothing here can catch that ahead of time - it
has to match byte-for-byte, including the trailing path).
