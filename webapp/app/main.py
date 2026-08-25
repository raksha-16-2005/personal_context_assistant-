"""The multi-tenant Gmail RAG backend. See ../../docs plan for the full
architecture; this wires the pieces built so far into one FastAPI app.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .account.routes import router as account_router
from .auth.routes import router as auth_router
from .calendar.routes import router as calendar_router
from .chat.routes import router as chat_router
from .db import init_schema
from .deps import get_settings
from .digest.routes import router as digest_router
from .pipeline_pool import PipelinePool

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_schema(settings.database_url)
    # Built once per process, not per request: this is the one loaded
    # embedding model every user's Pipeline shares - see pipeline_pool.py.
    pool = PipelinePool(
        settings.user_index_root, settings.shipped_chunking,
        settings.shipped_model, settings.shipped_rerank)
    # Blocking, on purpose: "Application startup complete" should mean the
    # server can actually answer a request fast, not that it's about to make
    # whichever user connects first pay for loading these from disk. A few
    # seconds added to startup is a better trade than a slow first /chat.
    # Skippable (see Settings.warm_pipeline_on_startup's own docstring) -
    # off by default in tests, where this pool is routinely replaced via
    # `get_pipeline_pool`'s dependency override before it's ever used.
    if settings.warm_pipeline_on_startup:
        pool.warm()
    app.state.pipeline_pool = pool

    # The job worker (syncs, extraction, digests) runs as a background
    # thread in this same process by default rather than a separate
    # deployment - see jobs/runner.py's own module docstring for why, and
    # Settings.run_worker_in_process for how to opt back into a standalone
    # `python -m app.jobs.runner` process instead. Sharing `pool` here is
    # what lets a completed sync invalidate that user's cached Pipeline
    # immediately (run_once's own pool.invalidate call).
    if settings.run_worker_in_process:
        from .jobs.runner import main_loop
        threading.Thread(
            target=main_loop, kwargs={"pool": pool}, daemon=True,
            name="job-worker").start()

    yield


app = FastAPI(title="Email RAG - multi-tenant backend", lifespan=lifespan)

# Only matters once the frontend is deployed separately from this backend
# (see Settings.frontend_base_url) - a same-origin deployment never sends a
# cross-origin request for the browser to preflight in the first place.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_base_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(account_router)
app.include_router(calendar_router)
app.include_router(digest_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Serves the built React app (see ../frontend) - guarded on the build
# actually existing so running the API alone (tests, `uvicorn` before
# anyone has run `npm run build`) is unaffected. Mounted last and matched
# last: every API route above still wins on an exact path match, so this
# only ever catches the frontend's own client-side routes (/chat,
# /commitments, ...), which is what lets a hard refresh on any of them still
# load the app instead of 404ing - a plain `StaticFiles(html=True)` only
# auto-serves index.html for a literal directory, not for a path with no
# file behind it.
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
             name="frontend-assets")

    @app.get("/favicon.svg", include_in_schema=False)
    def frontend_favicon():
        return FileResponse(FRONTEND_DIST / "favicon.svg")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend_catchall(full_path: str):
        return FileResponse(FRONTEND_DIST / "index.html")
