"""Postgres connection + schema bootstrap for the multi-tenant tables.

Same instance `index/store.py` already uses for pgvector (see
../config.yaml), different tables - `schema.sql` in this directory, not the
ablation `chunks_<config>` tables. Plain psycopg, no ORM: the rest of this
project already depends on psycopg directly (`index/store.py`), and an ORM
would be the only new abstraction in a codebase that otherwise reaches for
the plainest tool that works.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import psycopg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def connect(database_url: str):
    with psycopg.connect(database_url, autocommit=True) as conn:
        yield conn


def init_schema(database_url: str) -> None:
    """Create every table in schema.sql if it doesn't already exist.

    Idempotent (every statement is `CREATE TABLE IF NOT EXISTS` /
    `CREATE INDEX IF NOT EXISTS`), so this is safe to call on every process
    start rather than needing a separate migration-runner step for a project
    this size. Revisit with a real migration tool (Alembic) if the schema
    starts changing shape rather than just growing tables.
    """
    ddl = SCHEMA_PATH.read_text()
    with connect(database_url) as conn:
        conn.execute(ddl)
