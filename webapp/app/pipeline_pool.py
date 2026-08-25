"""Shared embedding model + a bounded per-user Pipeline cache.

This is the direct fix for the RAM-multiplication risk flagged in the plan:
a naive "one Pipeline per cached user" design would load a full copy of the
embedding model - hundreds of MB - per user, even though the weights are
identical for everyone (every mailbox is indexed with the one shipped
chunking/model config - see Settings.shipped_model/shipped_chunking). So the
model is loaded exactly once, here, and every cached `Pipeline` borrows it
through the `model=` param `Pipeline.__init__` was extended to accept. Ten
cached users now costs ten small index shards, not ten copies of the model.

Concurrency note, stated rather than hidden: `get()` holds one lock across
the whole lookup-or-construct path, including a cold `Pipeline` load. That
serializes cold loads across *all* users, not just one - correct and simple,
but a real ceiling if many users hit a cold cache at once. Worth revisiting
(e.g. a lock per user_id) if that ever shows up as latency; nothing here
needs it yet, matching the plan's own "Revisit later" items.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from emailrag.index import embed as E
from emailrag.index import rerank as RR
from emailrag.pipeline import Pipeline

from .ingestion.worker import bulk_messages_path, index_dir, messages_path

# Two, not build_index.py's single-tenant default of six: this process also
# answers live /chat requests and runs ingestion jobs concurrently, so one
# component must not claim every core.
POOL_THREADS = 2


class PipelinePool:
    def __init__(self, index_root: Path, chunking: str, model_id: str, rerank: str,
                max_cached: int = 8) -> None:
        self.index_root = Path(index_root)
        self.chunking = chunking
        self.model_id = model_id
        self.rerank = rerank
        self.max_cached = max_cached
        self._model = None
        self._reranker = None
        self._cache: "OrderedDict[str, Pipeline]" = OrderedDict()
        self._lock = threading.Lock()

    def _shared_model(self):
        if self._model is None:
            E.configure_threads(POOL_THREADS)
            self._model = E.load_model(self.model_id)
        return self._model

    def _shared_reranker(self):
        # Every shipped mailbox uses the same `rerank` spec (Settings.
        # shipped_rerank), so this cross-encoder's weights are identical for
        # every user too - the same RAM-multiplication reasoning this
        # module's docstring already makes for the embedding model, just
        # applied to the *other* model `Pipeline.__init__` loads. Before this,
        # every cold Pipeline load paid for its own copy.
        if self._reranker is None:
            spec = RR.SPECS.get(self.rerank)
            if spec is not None:
                self._reranker = RR.CrossEncoderReranker.from_spec(spec)
        return self._reranker

    def warm(self) -> None:
        """Load the shared embedding model and reranker now, in the
        server's own startup (see main.py's lifespan) rather than on
        whichever user's request happens to arrive first. Safe to call even
        when `rerank` has no matching spec (`_shared_reranker` just stays
        `None`, same as an unconfigured rerank would leave it per-Pipeline).
        """
        self._shared_model()
        self._shared_reranker()

    def get(self, user_id: str, conn=None) -> Pipeline:
        """`conn`, if given, is only used to rehydrate this user's index
        from Postgres (blob_store.download_user_root) when local disk
        doesn't already have it - a no-op whenever it does, i.e. every
        deployment with a persistent volume. See ingestion/blob_store.py."""
        with self._lock:
            cached = self._cache.pop(user_id, None)
            if cached is not None:
                self._cache[user_id] = cached          # mark most-recently-used
                return cached

            if conn is not None:
                from .ingestion.blob_store import download_user_root
                from .ingestion.worker import user_root
                download_user_root(conn, user_root(self.index_root, user_id), user_id)

            # `commitments=[]` here, not this user's real rows: the router
            # only needs to exist at construction time (this is where it's
            # built), the actual commitments list is refreshed on every
            # request instead - see chat/routes.py and
            # commitments.load_commitments_for_router's own docstring for why.
            pipe = Pipeline(
                index_dir(self.index_root, user_id), messages_path(self.index_root, user_id),
                rerank=self.rerank, model=self._shared_model(),
                reranker=self._shared_reranker(), verbose=False,
                route=True, commitments=[],
                bulk_sample=bulk_messages_path(self.index_root, user_id))

            self._cache[user_id] = pipe
            while len(self._cache) > self.max_cached:
                self._cache.popitem(last=False)        # drop the least-recently-used
            return pipe

    def invalidate(self, user_id: str) -> None:
        """Drop a cached Pipeline. Call after a sync job rebuilds this user's
        index, so the next /chat request opens the freshly written files
        instead of answering from a stale in-memory copy."""
        with self._lock:
            self._cache.pop(user_id, None)

    def __len__(self) -> int:
        return len(self._cache)
