from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# FastAPI's lifespan hook calls the real, `lru_cache`d get_settings() - which
# `app.dependency_overrides` cannot intercept, since that mechanism only
# overrides `Depends(...)`-injected callables at route resolution. Individual
# tests override settings for route *behaviour*; these env vars just have to
# exist so `load_settings()` never raises during app startup.
os.environ.setdefault("DATABASE_URL", "postgresql:///emailrag")
os.environ.setdefault("MASTER_KEY", Fernet.generate_key().decode())
os.environ.setdefault("SESSION_SECRET", "test-session-secret")
os.environ.setdefault("GMAIL_CLIENT_ID", "test-client-id")
os.environ.setdefault("GMAIL_CLIENT_SECRET", "test-client-secret")
# Every `TestClient(app)` construction re-runs the lifespan, and almost every
# test that uses one overrides `get_pipeline_pool` before making a real
# request - so the lifespan's own pool, and the ~14s `pool.warm()` would
# spend loading models into it, is never actually used. See
# app/config.py's Settings.warm_pipeline_on_startup for the real-deployment
# side of this.
os.environ.setdefault("WARM_PIPELINE_ON_STARTUP", "false")
# Same reasoning: a real background thread polling Postgres forever, started
# fresh by every TestClient(app) construction, would pile up across the test
# session and race tests that call jobs/runner.run_once directly.
os.environ.setdefault("RUN_WORKER_IN_PROCESS", "false")

from app.db import connect, init_schema

# The webapp's tables genuinely require Postgres - unlike the ML core, which
# keeps a DB optional so CI can run without one (see index/store.py's own
# lazy-import comment). Same local instance the ablation/pgvector work
# already uses.
TEST_DATABASE_URL = "postgresql:///emailrag"


@pytest.fixture(scope="session", autouse=True)
def _schema():
    init_schema(TEST_DATABASE_URL)


@pytest.fixture()
def db_conn():
    with connect(TEST_DATABASE_URL) as conn:
        yield conn


@pytest.fixture()
def test_user(db_conn):
    """A throwaway user row, deleted (cascading) after the test."""
    row = db_conn.execute(
        "INSERT INTO users (google_sub, email) VALUES (%s, %s) RETURNING id",
        (f"test-sub-{id(object())}", "test@example.com"),
    ).fetchone()
    user_id = row[0]
    yield str(user_id)
    db_conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
