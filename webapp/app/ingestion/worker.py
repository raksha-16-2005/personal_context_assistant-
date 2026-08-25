"""Per-user ingestion: Gmail -> dedup/filter -> thread -> chunk -> embed -> BM25.

Always rebuilds this user's chunk/embed/BM25 index from their full deduped
message set rather than patching it incrementally - simpler and correct, and
cheap enough at personal-mailbox scale (see the plan's latency section: a
typical mailbox is far smaller than the 50k-message benchmark corpus this
same chunk/embed/BM25 code already handles in minutes). A true incremental
re-embed - append-only DenseIndex/SparseIndex, chunk ids stable across runs -
is a real optimization, deliberately deferred: nothing here needs it yet, and
getting it wrong would silently corrupt a served index in a way a full
rebuild cannot.

Everything for one user lives under `<USER_INDEX_ROOT>/<user_id>/`:

    messages.parquet   deduped, filtered, threaded rows - MESSAGE_SCHEMA + thread_id
    sync_state.json    gmail.py's own incremental-sync cursor (historyId).
                       Kept as a local file rather than ported to Postgres like
                       DbTokenStore: `GmailClient` was already built to accept
                       any token-store-shaped object, so Postgres was a drop-in
                       swap there; `gmail.sync()` owns this file's read/write
                       directly and isn't parameterized the same way, so
                       teaching it a second backend isn't worth it yet. The
                       Postgres `sync_state` table instead mirrors *status*
                       ("still syncing" vs "ready") for the UI to poll cheaply
                       without touching this volume.
    index/             bm25/, dense/, chunks.jsonl, config.json - exactly the
                       shape scripts/build_index.py produces, so `Pipeline`
                       opens a user's index with zero code changes.

On a deployment with no persistent disk, this whole directory is only a
cache: blob_store.download_user_root() recreates it from Postgres before
this module reads any of it, and upload_user_root() saves a fresh copy
there right after each rebuild. A deployment with a real volume never
notices - download is a no-op once the directory already exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from emailrag.chunking import strategies as S
from emailrag.corpus import filters
from emailrag.corpus import gmail as G
from emailrag.corpus.schema import MESSAGE_SCHEMA
from emailrag.corpus.threads import assign_threads
from emailrag.index import chunktext as CT
from emailrag.index import embed as E
from emailrag.index.dense import DenseIndex
from emailrag.index.sparse import SparseIndex

from ..tokens.store import DbTokenStore
from .blob_store import download_user_root, upload_user_root

# Per-sync-job thread budget. Lower than build_index.py's single-tenant
# default (6): this runs on a box that also serves live /chat traffic and,
# via the job queue, other users' syncs - one job must not claim every core.
INGEST_THREADS = 2

# The first sync fetches only the last RECENT_SYNC_DAYS of mail, so a new
# user has something to chat with in about a minute instead of waiting for
# their entire history (measured at ~20 min for a real ~9k-message mailbox
# with sync's own concurrency - see corpus/gmail.py's FETCH_WORKERS). The
# rest of their mail arrives via a `backfill_history` job (jobs/runner.py),
# enqueued right after, using `before_query(RECENT_SYNC_DAYS)` as the
# complementary half of what the first pass already covered.
RECENT_SYNC_DAYS = 30


def user_root(index_root: Path, user_id: str) -> Path:
    return Path(index_root) / user_id


def messages_path(index_root: Path, user_id: str) -> Path:
    return user_root(index_root, user_id) / "messages.parquet"


def bulk_messages_path(index_root: Path, user_id: str) -> Path:
    return user_root(index_root, user_id) / "bulk_messages.parquet"


# Deliberately minimal, and deliberately not MESSAGE_SCHEMA: this exists only
# so "how many emails did I get today" can count what the bulk-mail filter
# (corpus/filters.is_bulk) already drops without a trace (see sync_user's own
# comment on this). No subject/body - the filter's whole point is that this
# mail never becomes searchable content, and duplicating it here would defeat
# that. `dedup_key`/`date_utc` is enough to count and window by arrival date;
# `sender` is enough for a note to name who it was from if that's ever useful.
# Webapp-only and never imported by scripts/build_corpus.py or Pipeline's
# required-file loading, so the CLI/eval corpus's shape is untouched.
BULK_SCHEMA = pa.schema([
    ("dedup_key", pa.string()),
    ("date_utc", pa.timestamp("us", tz="UTC")),
    ("sender", pa.string()),
])


def sync_state_path(index_root: Path, user_id: str) -> Path:
    return user_root(index_root, user_id) / "sync_state.json"


def index_dir(index_root: Path, user_id: str) -> Path:
    return user_root(index_root, user_id) / "index"


def _read_existing_dedup_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pq.read_table(path, columns=["dedup_key"]).column("dedup_key").to_pylist())


def _merge_messages(existing_path: Path, new_rows: list[dict]) -> list[dict]:
    """Everything already on disk for this user, plus newly fetched rows
    already deduped against it. Returned in memory rather than written here -
    the caller writes once, after re-threading, so `thread_id` is never
    briefly stale on disk between a merge and a rethread.
    """
    existing = pq.read_table(existing_path).to_pylist() if existing_path.exists() else []
    for row in existing:
        row.pop("thread_id", None)          # recomputed from scratch below
    return existing + new_rows


def _rethread(rows: list[dict]) -> list[dict]:
    thread_ids, _stats = assign_threads(rows)
    for row, tid in zip(rows, thread_ids):
        row["thread_id"] = tid
    return rows


def _write_messages(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [row.get(name) for row in rows] for name in MESSAGE_SCHEMA.names}
    table = pa.table(cols, schema=MESSAGE_SCHEMA)
    table = table.append_column(
        "thread_id", pa.array([row.get("thread_id", "") for row in rows], pa.string()))
    pq.write_table(table, path, compression="zstd")


def _new_bulk_rows(raw_messages: list[dict], already_seen: set[str]) -> list[dict]:
    """Which of this sync's raw fetch are bulk mail this file hasn't recorded
    yet - computed independently of `filters.dedup_and_filter`'s own pass
    over the same `raw_messages` (that call mutates its `seen` set in place to
    cover kept *and* bulk-dropped keys alike, so it can't be read afterward to
    tell the two apart). `already_seen` should be the union of the main
    corpus's existing keys and this file's own, both captured before that
    mutating call runs.
    """
    seen_in_batch: set[str] = set()
    out = []
    for msg in raw_messages:
        key = msg["dedup_key"]
        if key in already_seen or key in seen_in_batch:
            continue
        seen_in_batch.add(key)
        if filters.is_bulk(msg):
            out.append({"dedup_key": key, "date_utc": msg.get("date_utc"),
                       "sender": msg.get("sender") or ""})
    return out


def _write_bulk_messages(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [row.get(name) for row in rows] for name in BULK_SCHEMA.names}
    pq.write_table(pa.table(cols, schema=BULK_SCHEMA), path, compression="zstd")


def _progress_reporter(conn, user_id: str):
    """An `on_progress(completed, total)` callback (see corpus.gmail.sync's
    own docstring) that writes straight into sync_state, so /sync-status can
    compute a real ETA from real, in-flight numbers instead of a flat guess -
    `total` here is Gmail's own exact message count for this sync's query,
    not an estimate.
    """
    def report(completed: int, total: int) -> None:
        conn.execute(
            "UPDATE sync_state SET progress_current = %s, progress_total = %s "
            "WHERE user_id = %s", (completed, total, user_id))
    return report


def _rebuild_index(rows: list[dict], out_dir: Path, chunking: str, model_id: str,
                   shared_model=None) -> dict:
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    E.configure_threads(INGEST_THREADS)
    tokenizer = AutoTokenizer.from_pretrained(S.REFERENCE_TOKENIZER)
    chunks = S.chunk_corpus(chunking, tokenizer, rows)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]

    SparseIndex.build(ids, texts).save(out_dir / "bm25")

    # A caller serving many users shares one loaded model across every sync
    # job the same way PipelinePool shares it across every /chat request -
    # see Pipeline's own `model` param for the identical reasoning.
    model = shared_model if shared_model is not None else E.load_model(model_id)
    # `resume=False`, deliberately: `encode_corpus`'s checkpoint resume
    # assumes every found shard file holds exactly SHARD_SIZE rows and
    # advances its offset by that fixed amount regardless of the shard's
    # actual size - true only while re-running against the *same* `texts`
    # a killed job was interrupted on. This module's own docstring says the
    # opposite: every call here is a full rebuild of a mailbox whose size
    # changes between syncs, off a fixed `out_dir` that keeps yesterday's
    # shard files sitting on disk. A mailbox under one SHARD_SIZE (20,000
    # chunks - true for nearly every personal mailbox) writes just one
    # undersized "shard_0000000.npy"; the next sync's resume finds it,
    # advances past the *next* shard boundary as if it were full, and a
    # `range(start, n, SHARD_SIZE)` with `start` now past `n` silently
    # encodes nothing new at all - reproduced live: a mailbox grown from
    # 7,271 to 7,340 chunks came back with 7,340 ids and the old 7,271
    # vectors, and `DenseIndex` correctly refused to build a mismatched
    # index rather than silently drop the newest 69 chunks from search.
    # Skipping resume costs a few seconds of re-encoding a few thousand
    # short texts - the rebuild this function already always does - not the
    # 50k-message, hours-long run the checkpoint was built for.
    vectors = E.encode_corpus(model, texts, out_dir / "shards", batch_size=32,
                              resume=False)
    DenseIndex(ids, vectors).save(out_dir / "dense")

    with open(out_dir / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps({"chunk_id": c.chunk_id, "dedup_key": c.dedup_key,
                                 "thread_id": c.thread_id, "n_tokens": c.n_tokens}) + "\n")

    # Same stale-cache shape as the shard checkpoint above, one layer up:
    # `Pipeline.__init__` loads chunk texts via `CT.load_or_build`, which
    # trusts an existing `chunk_texts.jsonl.gz` beside the index only after
    # checking it against this run's fresh `chunks.jsonl` - and correctly
    # raises instead of silently reranking against yesterday's text when a
    # leftover cache from a smaller previous rebuild disagrees (reproduced
    # live, immediately after the fix above: a `ParityError` on the very
    # next /chat request, "7,340 local chunk ids vs 7,271"). Writing it
    # fresh here, from the `chunks` this rebuild already has in memory,
    # means there is never a stale cache to trip over - and skips
    # `load_or_build` re-deriving the same texts from `messages.parquet` on
    # every user's first request after a sync.
    CT.save_cache(out_dir / CT.CACHE_NAME,
                  CT.ChunkTexts(texts={c.chunk_id: c.text for c in chunks},
                               chunking=chunking))

    meta = {
        "chunking": chunking, "model": model_id, "dim": int(vectors.shape[1]),
        "n_messages": len(rows), "n_chunks": len(chunks),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    return meta


def sync_user(conn, settings, user_id: str, shared_model=None, query: str = "") -> dict:
    """Run one sync-and-reindex pass for `user_id`.

    `query` narrows what this pass fetches - empty means all history
    (incremental syncs, and a first sync with staging disabled), a recent
    window means the fast first pass `RECENT_SYNC_DAYS`/`backfill_history`
    are built around. Retrieval's *eventual* index still covers all history
    either way - `query` only changes how many passes it takes to get there.

    Raises on a Gmail/auth failure (an expired refresh token, a revoked
    grant) so the job queue's retry/give-up logic handles it rather than this
    function papering over it - `sync_state.status` is left at 'syncing' in
    that case, which the caller should flip to 'error' with the exception
    text (see app/jobs - the ingestion job handler, not this module, owns
    that decision, matching the queue's own attempts/last_error bookkeeping).
    """
    store = DbTokenStore(conn=conn, user_id=user_id, master_key=settings.master_key).load()
    client = G.GmailClient(settings.gmail_client_id, settings.gmail_client_secret,
                           tokens=store)

    state_path = sync_state_path(settings.user_index_root, user_id)
    msg_path = messages_path(settings.user_index_root, user_id)
    bulk_path = bulk_messages_path(settings.user_index_root, user_id)

    conn.execute(
        "UPDATE sync_state SET status = 'syncing', sync_started_at = now(), "
        "progress_current = 0, progress_total = 0 WHERE user_id = %s", (user_id,))

    # Only actually reads from Postgres when local disk doesn't already have
    # this user (see blob_store's own docstring) - a deployment with a
    # persistent volume never triggers it at all, since `root.exists()` is
    # already true. Without this, a wiped-disk incremental sync would read
    # zero existing dedup keys and treat every message as new.
    download_user_root(conn, user_root(settings.user_index_root, user_id), user_id)

    existing_keys = _read_existing_dedup_keys(msg_path)
    existing_bulk_keys = _read_existing_dedup_keys(bulk_path)
    raw_messages, gmail_state = G.sync(
        client, state_path, query=query, on_progress=_progress_reporter(conn, user_id))
    # `G.sync` returns the updated cursor but never writes it - by design,
    # since it's meant to work for any caller regardless of when that caller
    # considers a sync "done" (see that function's own docstring: "the
    # cursor is read *before* fetching"). Saving it is this caller's job, and
    # skipping it is why the very next sync always did a full re-fetch again
    # instead of ever actually going incremental: `state.history_id` loaded
    # empty every time because nothing had ever written it back to disk.
    gmail_state.save(state_path)

    # Computed before `dedup_and_filter` mutates `existing_keys` in place -
    # see `_new_bulk_rows`'s own docstring for why it can't be derived from
    # that call's aftermath instead.
    new_bulk = _new_bulk_rows(raw_messages, existing_keys | existing_bulk_keys)

    kept, filter_stats = filters.dedup_and_filter(raw_messages, seen=existing_keys)
    merged = _rethread(_merge_messages(msg_path, kept))
    _write_messages(msg_path, merged)

    if new_bulk:
        existing_bulk_rows = (pq.read_table(bulk_path).to_pylist()
                              if bulk_path.exists() else [])
        _write_bulk_messages(bulk_path, existing_bulk_rows + new_bulk)

    meta = _rebuild_index(merged, index_dir(settings.user_index_root, user_id),
                          settings.shipped_chunking, settings.shipped_model,
                          shared_model=shared_model)
    upload_user_root(conn, user_root(settings.user_index_root, user_id), user_id)

    conn.execute(
        "UPDATE sync_state SET status = 'ready', history_id = %s, "
        "last_sync_utc = %s, messages_seen = %s, error_detail = '' "
        "WHERE user_id = %s",
        (gmail_state.history_id, gmail_state.last_sync_utc,
         gmail_state.messages_seen, user_id))

    return {"new_messages": filter_stats["n_kept"], "new_bulk_messages": len(new_bulk),
           "total_messages": len(merged), **meta}


def backfill_history(conn, settings, user_id: str, shared_model=None) -> dict:
    """The one-time catch-up after a staged first sync: fetch everything
    from before `RECENT_SYNC_DAYS` ago, so a new user's index ends up
    covering their full mailbox without making them wait for it.

    Deliberately not `sync_user` with a different query - this never touches
    the Gmail `historyId` cursor at all (`sync_user`/`G.sync` own that), so a
    backfill racing a real incremental sync can't clobber it. Idempotent by
    the same mechanism every sync already relies on: `dedup_and_filter`
    against `existing_keys` skips anything this pass or an earlier one
    already wrote, so retrying a failed backfill is safe.
    """
    from emailrag.corpus.gmail import before_query

    store = DbTokenStore(conn=conn, user_id=user_id, master_key=settings.master_key).load()
    client = G.GmailClient(settings.gmail_client_id, settings.gmail_client_secret,
                           tokens=store)

    msg_path = messages_path(settings.user_index_root, user_id)
    bulk_path = bulk_messages_path(settings.user_index_root, user_id)
    download_user_root(conn, user_root(settings.user_index_root, user_id), user_id)
    existing_keys = _read_existing_dedup_keys(msg_path)
    existing_bulk_keys = _read_existing_dedup_keys(bulk_path)

    # Reuses the same progress columns sync_user's blocking wait does, even
    # though this runs after status is already 'ready' - /sync-status's
    # "still importing older mail" note (full_history_synced=false) reads
    # them too, just without an ETA calculation, since nothing is blocked on
    # this finishing.
    conn.execute(
        "UPDATE sync_state SET progress_current = 0, progress_total = 0 "
        "WHERE user_id = %s", (user_id,))
    raw_messages = G.fetch_messages(
        client, query=before_query(RECENT_SYNC_DAYS),
        on_progress=_progress_reporter(conn, user_id))
    new_bulk = _new_bulk_rows(raw_messages, existing_keys | existing_bulk_keys)

    kept, filter_stats = filters.dedup_and_filter(raw_messages, seen=existing_keys)
    merged = _rethread(_merge_messages(msg_path, kept))
    _write_messages(msg_path, merged)

    if new_bulk:
        existing_bulk_rows = (pq.read_table(bulk_path).to_pylist()
                              if bulk_path.exists() else [])
        _write_bulk_messages(bulk_path, existing_bulk_rows + new_bulk)

    meta = _rebuild_index(merged, index_dir(settings.user_index_root, user_id),
                          settings.shipped_chunking, settings.shipped_model,
                          shared_model=shared_model)
    upload_user_root(conn, user_root(settings.user_index_root, user_id), user_id)

    conn.execute(
        "UPDATE sync_state SET full_history_synced = true WHERE user_id = %s", (user_id,))

    return {"new_messages": filter_stats["n_kept"], "new_bulk_messages": len(new_bulk),
           "total_messages": len(merged), **meta}


# How stale a mailbox is allowed to get before the job runner refreshes it on
# its own. Gmail push notifications (Pub/Sub) would be the real answer and
# are deliberately not built yet - polling on the same cadence this queue
# already polls jobs at is the simplest thing that stops a mailbox going
# stale between logins, which used to be the *only* thing that ever
# re-synced a returning user: nothing anywhere else ever enqueues
# `incremental_sync` on its own.
SYNC_INTERVAL_MINUTES = 15


def schedule_due_syncs(conn) -> list[str]:
    """Enqueue `incremental_sync` for every ready user whose mailbox hasn't
    been refreshed in `SYNC_INTERVAL_MINUTES`, skipping anyone who already
    has one queued or running so this doesn't pile up duplicates on every
    poll - the same guard `digest.service.schedule_due_digests` uses for the
    same reason.
    """
    from ..jobs.queue import enqueue

    rows = conn.execute(
        """
        SELECT s.user_id FROM sync_state s
        WHERE s.status = 'ready'
          AND s.last_sync_utc <> ''
          AND s.last_sync_utc::timestamptz < now() - (%s * interval '1 minute')
          AND NOT EXISTS (
              SELECT 1 FROM jobs j
              WHERE j.user_id = s.user_id AND j.type = 'incremental_sync'
                AND j.status IN ('queued', 'running')
          )
        """,
        (SYNC_INTERVAL_MINUTES,),
    ).fetchall()

    user_ids = [str(r[0]) for r in rows]
    for user_id in user_ids:
        enqueue(conn, "incremental_sync", user_id=user_id)
    return user_ids
