"""Postgres-backed twin of emailrag.corpus.gmail.TokenStore.

Inherits every freshness/expiry computation from `TokenStoreBase` (the 7-day
Testing-refresh-token math, `access_token_valid`, `warn_if_expiring`) and
only implements where the bytes live: an encrypted row per user instead of a
0600 file on disk. `GmailClient` takes any `TokenStoreBase`-shaped object, so
it runs against this unmodified - see corpus/gmail.py's own docstring on
`TokenStoreBase` for why the split exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from emailrag.corpus.gmail import TokenStoreBase

from ..security import decrypt, encrypt


@dataclass
class DbTokenStore(TokenStoreBase):
    conn: object            # a psycopg connection, already open
    user_id: str
    master_key: str
    data: dict = field(default_factory=dict)

    def load(self) -> "DbTokenStore":
        row = self.conn.execute(
            "SELECT encrypted_refresh_token, encrypted_access_token, scope, "
            "extract(epoch from expires_at), extract(epoch from issued_at) "
            "FROM oauth_tokens WHERE user_id = %s",
            (self.user_id,),
        ).fetchone()
        if row is None:
            self.data = {}
            return self
        enc_refresh, enc_access, scope, expires_at, issued_at = row
        self.data = {
            "refresh_token": decrypt(self.master_key, bytes(enc_refresh)),
            "access_token": decrypt(self.master_key, bytes(enc_access)),
            "scope": scope,
            "expires_at": float(expires_at),
            "issued_at": float(issued_at),
        }
        return self

    def save(self) -> None:
        enc_refresh = encrypt(self.master_key, self.data.get("refresh_token", ""))
        enc_access = encrypt(self.master_key, self.data.get("access_token", ""))
        self.conn.execute(
            """
            INSERT INTO oauth_tokens
                (user_id, encrypted_refresh_token, encrypted_access_token,
                 scope, expires_at, issued_at)
            VALUES (%s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s))
            ON CONFLICT (user_id) DO UPDATE SET
                encrypted_refresh_token = EXCLUDED.encrypted_refresh_token,
                encrypted_access_token = EXCLUDED.encrypted_access_token,
                scope = EXCLUDED.scope,
                expires_at = EXCLUDED.expires_at,
                issued_at = EXCLUDED.issued_at
            """,
            (self.user_id, enc_refresh, enc_access,
             self.data.get("scope", ""), self.data.get("expires_at", 0),
             self.data.get("issued_at", 0)),
        )
