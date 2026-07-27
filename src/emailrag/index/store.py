"""pgvector store for the *served* system.

Scope note: the ablations do not go through Postgres (see `dense.py` - they
use exact search so the numbers isolate the embedding model). This store backs
the demo and the router, and it is where the approximate-search recall loss
gets measured against the exact baseline instead of being silently absorbed.

Two operational constraints from this machine, both handled here:

- `maintenance_work_mem` is raised **per session**, never with ALTER SYSTEM.
  This Postgres instance also hosts unrelated project databases and the HNSW
  build would otherwise starve them.
- Rows load via COPY. Row-by-row INSERT of 375k vectors is minutes of pure
  round-trip overhead.
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

_SAFE = re.compile(r"^[a-z0-9_]+$")


def table_name(chunking: str, model_id: str) -> str:
    """Deterministic, collision-free table name for one ablation config."""
    model_slug = model_id.split("/")[-1].replace("-", "_").replace(".", "_").lower()
    name = f"chunks_{chunking}__{model_slug}"
    if not _SAFE.match(name):
        raise ValueError(f"unsafe table name derived: {name!r}")
    return name


@contextmanager
def connect(dsn: str, maintenance_work_mem: str | None = None):
    with psycopg.connect(dsn, autocommit=True) as conn:
        register_vector(conn)
        if maintenance_work_mem:
            # SET LOCAL would need a transaction; a plain SET is session-scoped,
            # which is what we want and still never touches the global config.
            conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
        yield conn


def create_table(conn, table: str, dim: int, drop: bool = False) -> None:
    if not _SAFE.match(table):
        raise ValueError(f"unsafe table name: {table!r}")
    if drop:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            chunk_id   text PRIMARY KEY,
            dedup_key  text NOT NULL,
            thread_id  text,
            ordinal    int,
            n_tokens   int,
            text       text,
            embedding  vector({dim})
        )
    """)
    # dedup_key is the join key for collapsing chunk hits to message hits.
    conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_dedup_idx ON {table} (dedup_key)")


def copy_chunks(conn, table: str, chunks: list, embeddings: np.ndarray) -> int:
    """Bulk-load chunks with their vectors. `chunks[i]` must match `embeddings[i]`."""
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"{len(chunks)} chunks vs {embeddings.shape[0]} vectors")

    cols = "(chunk_id, dedup_key, thread_id, ordinal, n_tokens, text, embedding)"
    with conn.cursor().copy(f"COPY {table} {cols} FROM STDIN") as copy:
        for chunk, vec in zip(chunks, embeddings):
            copy.write_row((
                chunk.chunk_id, chunk.dedup_key, chunk.thread_id,
                chunk.ordinal, chunk.n_tokens, chunk.text,
                # pgvector's text input format; COPY is text-mode here.
                "[" + ",".join(f"{v:.7g}" for v in vec) + "]",
            ))
    return len(chunks)


def build_hnsw(conn, table: str, m: int = 16, ef_construction: int = 64) -> None:
    """Build the ANN index.

    Uses `vector_ip_ops` (inner product) rather than `vector_cosine_ops`:
    embeddings are L2-normalized at encode time, so inner product and cosine
    rank identically and inner product skips the norm computation per
    comparison. Build after loading - incremental insert into an existing HNSW
    is far slower.
    """
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {table}_hnsw
        ON {table} USING hnsw (embedding vector_ip_ops)
        WITH (m = {m}, ef_construction = {ef_construction})
    """)
    conn.execute(f"ANALYZE {table}")


def search(conn, table: str, query_vec: np.ndarray, top_k: int = 100,
           ef_search: int = 100) -> list[tuple[str, float]]:
    """Top-k by cosine. `ef_search` trades recall against latency at query time."""
    conn.execute(f"SET hnsw.ef_search = {ef_search}")
    # '<#>' is negative inner product, so negating restores a similarity where
    # larger is better - matching what DenseIndex returns.
    rows = conn.execute(
        f"SELECT chunk_id, -(embedding <#> %s) AS score "
        f"FROM {table} ORDER BY embedding <#> %s LIMIT %s",
        (query_vec, query_vec, top_k),
    ).fetchall()
    return [(cid, float(score)) for cid, score in rows]
