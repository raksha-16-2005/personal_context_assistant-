# Deploying to Fly.io

One-time setup, then a normal `fly deploy` for every update after. Run
these from the repo root (same directory as `Dockerfile` and `fly.toml`).

## 0. Install flyctl and log in

```bash
brew install flyctl
fly auth login          # opens a browser - this step needs you
```

## 1. Create the app (picks your unique name)

```bash
fly apps create           # interactive - suggests a name, or type your own
```

Whatever name it gives you, put it in **two** places:
- `fly.toml`'s `app = "..."` line
- `fly.toml`'s `OAUTH_REDIRECT_BASE_URL = "https://<name>.fly.dev"` line

## 2. Postgres

```bash
fly postgres create --name <name>-db     # pick the free/hobby tier when asked
fly postgres attach <name>-db --app <name>
```

`attach` sets `DATABASE_URL` as a secret automatically - you don't set it
by hand.

## 3. Persistent volume (for USER_INDEX_ROOT - per-user mailbox indices)

```bash
fly volumes create emailrag_data --app <name> --region iad --size 3
```

Match `--region` to `fly.toml`'s `primary_region`.

## 4. Secrets (never in fly.toml - these are the ones that matter if leaked)

```bash
fly secrets set --app <name> \
  MASTER_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" \
  SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  GMAIL_CLIENT_ID="<from Google Cloud Console>" \
  GMAIL_CLIENT_SECRET="<from Google Cloud Console>"
```

## 5. Google OAuth client - redirect URI must match the real URL

Back in Google Cloud Console (APIs & Services -> Credentials -> your OAuth
client -> Authorized redirect URIs), add:

```
https://<name>.fly.dev/auth/google/callback
```

(`localhost:8000` can stay too, if you still want local testing to work.)

## 6. Deploy

```bash
fly deploy
```

This runs `release_command` (`python -m app.security` - refuses to roll out
if any stored secret isn't real ciphertext, see that module's docstring)
before traffic ever reaches the new version, then starts the `app` and
`worker` process groups defined in `fly.toml`.

The `worker` process group (the job runner - syncs, extraction, digests) is
defined but Fly does not auto-scale a count for non-HTTP process groups on
first deploy. After the first successful deploy:

```bash
fly scale count app=1 worker=1 --app <name>
```

## 7. Check it

```bash
fly status --app <name>
curl https://<name>.fly.dev/healthz
open https://<name>.fly.dev
```

## Updating after the first deploy

Just `fly deploy` again - it rebuilds the image, runs the release command,
and does a rolling replace of the running machines.

## If something won't start

```bash
fly logs --app <name>
```

The most likely first-deploy failures: a secret typo'd or missing
(`load_settings()` raises `ConfigError` naming the exact key), or the OAuth
redirect URI not matching exactly (Google returns `redirect_uri_mismatch`
and nothing here can catch that ahead of time - it has to match
byte-for-byte, including the trailing path).
