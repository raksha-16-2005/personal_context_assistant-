"""Settings, read from webapp/.env the same way the rest of this project reads
../.env - see emailrag.llm.client.load_dotenv, reused rather than reinvented.

A dataclass, not a global dict of `os.environ.get(...)` calls scattered across
every module: every required setting is named once, here, and a missing one
fails at startup with a message that says which .env key to set - the same
"exact setup step, not a stack trace" philosophy as `llm.client.MissingKey`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

WEBAPP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEBAPP_ROOT.parent

sys.path.insert(0, str(REPO_ROOT / "src"))

from emailrag.llm.client import load_dotenv  # noqa: E402


class ConfigError(RuntimeError):
    """Raised with the exact .env key missing, not a stack trace."""


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(
            f"{name} is not set.\n"
            f"  cp webapp/.env.example webapp/.env, then fill it in.")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    master_key: str
    session_secret: str
    gmail_client_id: str
    gmail_client_secret: str
    oauth_redirect_base_url: str
    user_index_root: Path
    shipped_chunking: str
    shipped_model: str
    shipped_rerank: str
    # Off in tests (see conftest.py) - real deployments want the embedding
    # model and reranker loaded before the first request, but a test that
    # overrides `get_pipeline_pool` (most of them) never touches the
    # lifespan's own pool at all, so warming it is ~14s spent on a pool
    # nothing will ever use. Every `TestClient(app)` construction re-runs
    # the lifespan, so that cost was being paid once per test function.
    warm_pipeline_on_startup: bool = True

    @property
    def oauth_redirect_uri(self) -> str:
        return f"{self.oauth_redirect_base_url.rstrip('/')}/auth/google/callback"


def load_settings() -> Settings:
    load_dotenv(WEBAPP_ROOT / ".env")
    return Settings(
        database_url=_require("DATABASE_URL"),
        master_key=_require("MASTER_KEY"),
        session_secret=_require("SESSION_SECRET"),
        gmail_client_id=_require("GMAIL_CLIENT_ID"),
        gmail_client_secret=_require("GMAIL_CLIENT_SECRET"),
        oauth_redirect_base_url=os.environ.get(
            "OAUTH_REDIRECT_BASE_URL", "http://localhost:8000"),
        user_index_root=Path(os.environ.get(
            "USER_INDEX_ROOT", str(REPO_ROOT / "data" / "index" / "users"))),
        shipped_chunking=os.environ.get("SHIPPED_CHUNKING", "thread_aware"),
        shipped_model=os.environ.get(
            "SHIPPED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        shipped_rerank=os.environ.get("SHIPPED_RERANK", "L2@20/t192"),
        warm_pipeline_on_startup=os.environ.get(
            "WARM_PIPELINE_ON_STARTUP", "true").lower() not in ("false", "0", ""),
    )
