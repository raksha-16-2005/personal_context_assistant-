"""Durable backing store for a user's on-disk index directory, for a
deployment with no persistent disk (e.g. Render's free tier, whose
filesystem is wiped on every redeploy and every wake from its 15-minute
idle sleep). Local disk stays exactly what worker.py and Pipeline already
read/write - this only adds a way to recreate it from Postgres when it's
gone, and to save a copy there once a sync finishes.

One row per user, one blob: everything under `user_root(index_root,
user_id)` (messages.parquet, bulk_messages.parquet, sync_state.json,
index/) tar+gzipped into a single `bytea`. Measured at ~30-40 MB for a
typical mailbox (see the deploy notes) - comfortably a `bytea`/TOAST value,
not something that needs its own object storage.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

# Takes the user's root directory directly rather than importing worker.py's
# user_root() to compute it - worker.py is this module's only caller and
# already has that path, and importing back from it here would make the two
# modules import each other.
#
# encode_corpus's own checkpoint scratch space (webapp/app/ingestion/
# worker.py's _rebuild_index) - never read again once a rebuild finishes,
# and never cleaned up locally between syncs. Excluding it here is what
# stops that local-disk leak from also inflating every blob this writes.
_EXCLUDE_DIRNAMES = {"shards"}


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if any(part in _EXCLUDE_DIRNAMES for part in Path(tarinfo.name).parts):
        return None
    return tarinfo


def upload_user_root(conn, root: Path, user_id: str) -> None:
    """Save this user's whole on-disk directory (root = worker.user_root(...))
    as the new durable copy. Called right after a sync rebuilds it, so the
    row in Postgres is never more than one sync behind local disk."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root, arcname=".", filter=_tar_filter)
    conn.execute(
        "INSERT INTO user_index_blobs (user_id, data, updated_at) "
        "VALUES (%s, %s, now()) "
        "ON CONFLICT (user_id) DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
        (user_id, buf.getvalue()))


def download_user_root(conn, root: Path, user_id: str) -> bool:
    """Recreate this user's on-disk directory from the last saved blob, if
    local disk doesn't already have it. Returns whether there was a blob to
    restore - False means either local disk is already there (nothing to
    do) or this is a genuinely new user with nothing synced yet, not a
    wiped cache."""
    if root.exists():
        return False
    row = conn.execute(
        "SELECT data FROM user_index_blobs WHERE user_id = %s", (user_id,)).fetchone()
    if row is None:
        return False
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(row[0]), mode="r:gz") as tar:
        tar.extractall(root, filter="data")
    return True
