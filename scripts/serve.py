#!/usr/bin/env python
"""Local web UI for the assembled system.

    make serve                      # then open http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --rerank none

Built on `http.server` from the standard library, with no new dependencies, and
that is a deliberate choice rather than minimalism for its own sake. This
project's `requirements.txt` is pinned to the newest combination of
torch/transformers/sentence-transformers that works on Intel macOS (torch 2.2.2
is the last version with a `macosx_x86_64` wheel). Gradio and Streamlit both
require newer `huggingface_hub` and `pydantic` than that stack tolerates, so
adding either would trade a working retrieval pipeline for a nicer widget set.
The public demo Space in phase 6 is where Gradio belongs - in a fresh
environment, on ZeroGPU, where the pin does not apply.

Two things this server does not do, on purpose:

**It binds to 127.0.0.1 only.** Phase 7 points this at a real mailbox. A local
tool that answers questions about private mail must not be one `--host 0.0.0.0`
away from serving it to the network, so the bind address is not a flag.

**It loads the index once, at startup.** The dense matrix is ~90 MB, chunk texts
~65 MB, and the first encode costs ~1.3 s of lazy initialisation. Per-request
loading would make the UI unusable and would also make every latency number it
displays a lie.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index import rerank as RR  # noqa: E402
from emailrag.pipeline import DEFAULT_RERANK, Pipeline  # noqa: E402
from emailrag.query.transform import QueryTransformer  # noqa: E402

UI_DIR = Path(__file__).resolve().parents[1] / "src" / "emailrag" / "ui"

# Inference is serialised. torch is configured for 6 intra-op threads (physical
# cores), so two concurrent requests would not go faster - they would contend for
# the same AVX2 units and make both slower while corrupting the timings the page
# displays. One request at a time, honestly measured.
INFERENCE_LOCK = threading.Lock()

MAX_QUESTION_CHARS = 500


class Handler(BaseHTTPRequestHandler):
    pipeline: Pipeline = None            # set on the class before serving
    server_version = "emailrag/0.1"

    def log_message(self, fmt: str, *args) -> None:
        # The default logs every asset request; only queries are interesting.
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write(f"  {fmt % args}\n")

    # -- plumbing ----------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # No external resources are referenced, and none should be loadable: this
        # page will eventually render private mail.
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; "
                         "script-src 'unsafe-inline'; connect-src 'self'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes ------------------------------------------------------------

    @property
    def route(self) -> str:
        """Path without the query string.

        `self.path` includes it, so matching on `self.path` directly 404s the
        deep links the UI writes (`/?q=...`) - which is every bookmarked or
        shared result.
        """
        from urllib.parse import urlsplit
        return urlsplit(self.path).path

    def do_GET(self) -> None:
        if self.route in ("/", "/index.html"):
            self._send(200, (UI_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.route == "/api/config":
            self._json({
                **self.pipeline.config_summary,
                "rerank_arms": sorted(RR.SPECS),
                "transforms": list(QueryTransformer.KINDS),
                "retrievers": ["hybrid_rrf", "hybrid_weighted", "dense", "bm25"],
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.route not in ("/api/ask", "/api/search"):
            self._json({"error": "not found"}, 404)
            return
        try:
            body = self._read_json()
        except json.JSONDecodeError:
            self._json({"error": "malformed JSON"}, 400)
            return

        question = (body.get("question") or "").strip()
        if not question:
            self._json({"error": "empty question"}, 400)
            return
        if len(question) > MAX_QUESTION_CHARS:
            self._json({"error": f"question over {MAX_QUESTION_CHARS} characters"}, 400)
            return

        pipe = self.pipeline
        # Per-request overrides. Reranker and transform are cheap to swap; the
        # index and embedding model are not, so those stay fixed for the process.
        with INFERENCE_LOCK:
            try:
                self._apply_overrides(body)
            except ValueError as exc:
                # An unknown arm is the caller's mistake, not the server's.
                self._json({"error": str(exc)}, 400)
                return
            try:
                t0 = time.perf_counter()
                if self.route == "/api/search":
                    result = pipe.search(question, top_n=int(body.get("top_n") or 10),
                                         as_of=body.get("as_of") or "")
                    payload = {"answer": None,
                               "search": _search_json(result),
                               "wall_ms": (time.perf_counter() - t0) * 1000}
                else:
                    answer, result = pipe.ask(
                        question, n_sources=int(body.get("n_sources") or 6),
                        as_of=body.get("as_of") or "",
                        top_n=int(body.get("top_n") or 10))
                    payload = {"answer": _answer_json(answer),
                               "search": _search_json(result),
                               "wall_ms": (time.perf_counter() - t0) * 1000}
            except Exception as exc:                      # noqa: BLE001
                # A failed query must not take the server down - the index took
                # ten seconds to load and the next question may well work.
                import traceback
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return

        payload["config"] = pipe.config_summary
        self._json(payload)

    def _apply_overrides(self, body: dict) -> None:
        pipe = self.pipeline
        retriever = body.get("retriever")
        if retriever and retriever != pipe.retriever:
            if retriever not in ("bm25", "dense", "hybrid_rrf", "hybrid_weighted"):
                raise ValueError(f"unknown retriever {retriever!r}")
            pipe.retriever = retriever

        rerank = body.get("rerank")
        if rerank is not None and rerank != pipe.rerank:
            if rerank not in RR.SPECS:
                raise ValueError(f"unknown rerank arm {rerank!r}")
            spec = RR.SPECS[rerank]
            pipe.rerank = rerank
            pipe.reranker = (RR.CrossEncoderReranker.from_spec(spec) if spec else None)
            pipe.rerank_top_k = spec.top_k if spec else 0

        transform = body.get("transform")
        if transform is not None and transform != pipe.transform:
            if transform not in QueryTransformer.KINDS:
                raise ValueError(f"unknown transform {transform!r}")
            pipe.transform = transform
            pipe.transformer = (QueryTransformer(transform) if transform != "none"
                                else None)


def _search_json(result) -> dict:
    return {
        "question": result.question,
        "timings": {k: round(v, 1) for k, v in result.timings.items()},
        "transform_kind": result.transform_kind,
        "transform_texts": result.transform_texts,
        "transform_degraded": result.transform_degraded,
        "messages": [{
            "n": i + 1,
            "message_id": m.message_id,
            "sender": m.sender,
            "recipients": m.recipients,
            "date": m.date,
            "subject": m.subject,
            "score": round(m.score, 4),
            "n_chunks": len(m.chunk_ids),
            "text": m.text,
        } for i, m in enumerate(result.messages)],
    }


def _answer_json(answer) -> dict:
    return {
        "text": answer.text,
        "refused": answer.refused,
        "cited": answer.cited_numbers,
        "invalid_citations": answer.invalid_citations,
        "uncited_sentences": answer.uncited_sentences,
        "grounded_shape": answer.is_grounded_shape,
        "model": answer.model,
        "cached": answer.cached,
        "latency_ms": round(answer.latency_ms, 1),
        "error": answer.error,
    }


def default_index(root: Path) -> Path | None:
    built = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "config.json").exists()) if root.exists() else []
    return built[0] if built else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--index-root", type=Path, default=Path("data/index"))
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--retriever", default="hybrid_rrf")
    ap.add_argument("--rerank", default=DEFAULT_RERANK)
    ap.add_argument("--transform", default="none")
    args = ap.parse_args()

    index = args.index or default_index(args.index_root)
    if index is None:
        print(f"error: no built index under {args.index_root}.\n"
              f"  build one: make index-pilot", file=sys.stderr)
        return 1
    if not args.sample.exists():
        print(f"error: {args.sample} missing - run `make sample`", file=sys.stderr)
        return 1

    Handler.pipeline = Pipeline(index, args.sample, retriever=args.retriever,
                                rerank=args.rerank, transform=args.transform)

    # 127.0.0.1, never 0.0.0.0, and not configurable - see the module docstring.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  serving on http://127.0.0.1:{args.port}  (ctrl-c to stop)")
    print(f"  {Handler.pipeline.config_summary}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
