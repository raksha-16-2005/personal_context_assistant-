#!/usr/bin/env python
"""One-time (well, weekly) Gmail authorisation.

    python scripts/gmail_auth.py            # authorise
    python scripts/gmail_auth.py --status   # how long the token has left

**Weekly, not one-time, and that is Google's design rather than an oversight
here.** An OAuth app in Testing status gets refresh tokens that expire after 7
days. The long-lived-token exception covers basic profile scopes only - not Gmail.
Escaping it requires Google app verification, which for a Gmail scope means a
security review. So personal use means re-running this weekly, and `--status`
exists so that is a calendar reminder rather than a surprise mid-sync.

The loopback flow runs a throwaway HTTP server on localhost to catch the redirect,
which is the flow Google documents for desktop apps. Nothing is sent anywhere
except Google: the code is exchanged for tokens directly, and the tokens land in
~/.config/emailrag/ with mode 0600 - outside the repo, so no `.gitignore` mistake
can publish them.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.corpus import gmail as G  # noqa: E402
from emailrag.llm.client import load_dotenv  # noqa: E402

PORT = 8765
REDIRECT = f"http://localhost:{PORT}/"


class _CatchCode(BaseHTTPRequestHandler):
    code: str = ""
    error: str = ""

    def do_GET(self) -> None:
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        _CatchCode.code = (params.get("code") or [""])[0]
        _CatchCode.error = (params.get("error") or [""])[0]

        body = (b"<h2>Authorised.</h2><p>You can close this tab.</p>"
                if _CatchCode.code else
                f"<h2>Authorisation failed</h2><p>{_CatchCode.error}</p>".encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:
        pass          # the redirect URL contains the auth code; do not log it


def show_status(store: G.TokenStore) -> int:
    if not store.refresh_token:
        print("no token stored. Run without --status to authorise.")
        return 1
    left = store.days_until_reauthorization()
    print(f"token file      {store.path}")
    print(f"refresh token   {store.refresh_token_age_days():.1f} days old")
    print(f"expires in      {left:.1f} days "
          f"(Testing-status apps get {G.TESTING_REFRESH_TOKEN_DAYS} days for "
          f"Gmail scopes)")
    print(f"access token    "
          f"{'valid' if store.access_token_valid() else 'expired - will refresh'}")
    warning = store.warn_if_expiring()
    if warning:
        print(f"\n{warning}")
    return 0 if left > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--token-path", type=Path, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    store = G.TokenStore(path=args.token_path or G.DEFAULT_TOKEN_PATH).load()

    if args.status:
        return show_status(store)

    import os
    client_id = os.environ.get("GMAIL_CLIENT_ID", "")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print("error: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET are not set.\n"
              "  1. Google Cloud console -> new project\n"
              "  2. Enable the Gmail API\n"
              "  3. OAuth consent screen -> External, status Testing, add yourself\n"
              "     as a test user\n"
              "  4. Credentials -> OAuth client ID -> Desktop app\n"
              f"  5. Add {REDIRECT} as an authorised redirect URI\n"
              "  6. Put both values in .env (gitignored)", file=sys.stderr)
        return 1

    url = G.authorization_url(client_id, REDIRECT)
    print(f"opening the consent screen. If it does not open, visit:\n\n{url}\n")
    if not args.no_browser:
        webbrowser.open(url)

    print(f"waiting for the redirect on {REDIRECT} ...")
    server = HTTPServer(("127.0.0.1", PORT), _CatchCode)
    server.timeout = 300
    server.handle_request()
    server.server_close()

    if not _CatchCode.code:
        print(f"error: no authorisation code received "
              f"({_CatchCode.error or 'timed out'})", file=sys.stderr)
        return 1

    print("exchanging the code for tokens ...")
    tokens = G.exchange_code(client_id, client_secret, _CatchCode.code, REDIRECT)
    if not tokens.get("refresh_token"):
        print("error: Google returned no refresh token. This happens when the app "
              "has been authorised before without prompt=consent; revoke access at "
              "https://myaccount.google.com/permissions and retry.", file=sys.stderr)
        return 1

    store.data.update({
        "refresh_token": tokens["refresh_token"],
        "access_token": tokens.get("access_token", ""),
        "expires_at": time.time() + int(tokens.get("expires_in", 3600)),
        "issued_at": time.time(),
        "scope": tokens.get("scope", G.SCOPE),
    })
    store.save()
    print(f"\nwrote {store.path} (mode 0600, outside the repo)")

    client = G.GmailClient(client_id, client_secret,
                           tokens=G.TokenStore(path=store.path))
    profile = client.profile()
    print(f"authorised as {profile.get('emailAddress')} "
          f"({profile.get('messagesTotal', 0):,} messages)")
    print(f"\nRe-run this in {G.TESTING_REFRESH_TOKEN_DAYS} days - Testing-status "
          f"apps do not get long-lived refresh tokens for Gmail scopes.")
    print("Check any time with: python scripts/gmail_auth.py --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
