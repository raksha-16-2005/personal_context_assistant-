"""One question in, a cited answer out - the assembled system.

The ablation harness (`evaluation/harness.py`) and this module solve different
problems and are deliberately separate. The harness runs 80 queries through many
configurations and reports metrics; it holds no message metadata, because metrics
never need it. This runs *one* question through *one* configuration and has to
produce something a person can read: sender, date, subject, the excerpt, and a
link back to the message. Sharing one code path would mean either loading message
metadata for 80x6x4 benchmark rows that ignore it, or shipping a UI that can only
display chunk ids.

What they do share is every component - the same index, the same fusion, the same
reranker, the same transforms - so a configuration that wins the ablation is the
configuration this serves. The `RunConfig` names are the same strings.

Loading is lazy and cached per process: the dense matrix is ~90 MB, the chunk
texts ~65 MB, and a UI that reloaded them per request would be unusable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .generation.synthesize import Answer, Citation, Synthesizer
from .index import chunktext as CT
from .index import embed as E
from .index import rerank as RR
from .index.dense import DenseIndex
from .index.fusion import reciprocal_rank_fusion, weighted_score_fusion
from .index.sparse import SparseIndex
from .llm.client import LLM
from .query.transform import QueryTransformer

# Chunks pulled before collapsing to messages. Deeper than the number of sources
# shown, for the same reason the harness retrieves deep: several chunks routinely
# belong to one message, so 60 chunks can be far fewer than 60 messages.
RETRIEVE_DEPTH = 120

DEFAULT_RERANK = "L2@20/t192"      # the only arm inside the 200 ms CPU budget


@dataclass(slots=True)
class RetrievedMessage:
    message_id: str
    score: float
    rank: int
    chunk_ids: list[str] = field(default_factory=list)
    text: str = ""
    sender: str = ""
    recipients: str = ""
    date: str = ""
    subject: str = ""


@dataclass(slots=True)
class SearchResult:
    question: str
    messages: list[RetrievedMessage]
    timings: dict = field(default_factory=dict)
    transform_kind: str = "none"
    transform_texts: list[str] = field(default_factory=list)
    transform_degraded: bool = False
    # The routing decision and, when the SQL arm ran, what it returned. `None`
    # means no router is configured and retrieval ran unconditionally.
    route: dict | None = None
    commitments: list[dict] = field(default_factory=list)


def _corpus_end_date(dates: list, percentile: float = 0.99) -> str:
    """The corpus's effective end date, as the default `as_of` anchor.

    A percentile, not `max()`, because Enron's headers contain garbage dates: the
    50k sample spans 1980-01-01 to **2044-01-04**, with 19 messages after 2003 and
    88 before 1998 against a real span of 1999-2002. Anchoring "what's due next
    week" on the maximum would put the window in 2044, every temporal query would
    return nothing, and the emptiness would read as a retrieval failure rather than
    as one malformed Date header.

    This is only a fallback. Temporal eval queries carry their own `as_of` for
    exactly this reason, and a caller that knows the date should pass it.
    """
    if not dates:
        return ""
    ordered = sorted(dates)
    index = min(int(len(ordered) * percentile), len(ordered) - 1)
    return ordered[index].strftime("%Y-%m-%d")


def collapse(ranked: list[tuple[str, float]], top_n: int,
             texts: dict[str, str], headers: dict[str, dict]
             ) -> list[RetrievedMessage]:
    """Chunk ranking -> message ranking, keeping each message's best position.

    A module-level function rather than a method because it is the part of this
    file with rules worth testing, and it needs nothing from the 155 MB of loaded
    index to do its job.

    A message's several retrieved chunks are concatenated in rank order, so a
    citation shows everything retrieval found in that message rather than an
    arbitrary one of its fragments. `top_n` counts *messages*, not chunks - the
    caller asked for ten things to read, not ten fragments of three things.
    """
    order: list[str] = []
    by_message: dict[str, RetrievedMessage] = {}

    for chunk_id, score in ranked:
        key = chunk_id.rsplit(":", 1)[0]
        if key not in by_message:
            if len(by_message) >= top_n:
                # Full on messages, but later chunks of messages already in the
                # list still get appended below - dropping them would show a
                # partial excerpt of a message we retrieved twice.
                continue
            head = headers.get(key, {})
            by_message[key] = RetrievedMessage(
                message_id=key, score=float(score), rank=len(order) + 1,
                sender=head.get("sender", ""),
                recipients=head.get("recipients", ""),
                date=head.get("date", ""),
                subject=head.get("subject", ""),
            )
            order.append(key)
        msg = by_message[key]
        msg.chunk_ids.append(chunk_id)
        piece = texts.get(chunk_id, "")
        if piece and piece not in msg.text:
            msg.text = f"{msg.text}\n{piece}".strip() if msg.text else piece

    return [by_message[k] for k in order]


def _commitment_row_to_message(row: dict, rank: int, messages: dict) -> RetrievedMessage:
    """A `_sql_arm` hit, shaped like something `ask()` can cite.

    There is no chunk text to show - a commitment is a fact extracted from a
    message, not a retrieved passage - so `text`/`subject` are the
    commitment's own words instead. `score` is the extractor's confidence,
    not a similarity score; the two are not comparable, but both are already
    "higher is better", which is all sorting needs.

    `date` is looked up from `messages` (the *source* message's real
    received date) rather than `row["due_at"]` on purpose. Showing the due
    date as "Date:" told the generator this arrived on its deadline, which
    is a different fact from when it actually landed - a commitment due
    today can come from a message that arrived days ago, and reporting the
    due date as the received date is exactly what turned "you have a
    deadline today" into the wrong claim "you received an email today."
    Kept in the citation's `text` field (via `due_phrase`) instead, where a
    due date belongs.

    The "Deadline:" prefix is what tells the generator (and a reader of the
    citation itself) that this is an *extracted obligation*, not the
    message's real subject line - without it, a question that asks for both
    "what's due" and "a summary of my mail" has no signal in the citation
    that would stop it blending the two into one claim.
    """
    source = messages.get(row["message_id"], {})
    text = row["text"]
    if row.get("due_phrase"):
        text = f"{text} (due {row['due_phrase']})"
    return RetrievedMessage(
        message_id=row["message_id"], score=row.get("confidence", 0.0), rank=rank,
        text=text,
        sender=row.get("owner") or source.get("sender", ""),
        recipients=row.get("counterparty") or source.get("recipients", ""),
        date=source.get("date", ""),
        subject=f"Deadline: {row['text'][:110]}",
    )


def _message_date_row_to_message(row: dict, rank: int) -> RetrievedMessage:
    """A `_message_date_arm` hit. Subject-only, not a body excerpt: this is a
    date-scoped listing ("what came in today"), not a similarity match, and
    there is no ranked chunk to justify preferring one part of the body over
    another - the subject is what a real inbox listing would show anyway.
    """
    return RetrievedMessage(
        message_id=row["dedup_key"], score=1.0, rank=rank,
        text=row.get("subject", ""), sender=row.get("sender", ""),
        recipients=row.get("recipients", ""), date=row.get("date", ""),
        subject=row.get("subject", ""),
    )


class Pipeline:
    """A loaded index plus the optional reranker and transform, ready to query."""

    def __init__(self, index_dir: Path, sample: Path, *, retriever: str = "hybrid_rrf",
                 rerank: str = DEFAULT_RERANK, transform: str = "none",
                 dense_weight: float = 0.5, threads: int = 6,
                 route: bool = False, commitments: Path | list | None = None,
                 verbose: bool = True, model=None, reranker=None,
                 llm: LLM | None = None, bulk_sample: Path | None = None) -> None:
        """`model` and `reranker` let a caller serving many users from one process
        (the multi-tenant web app's `PipelinePool`) share one loaded embedding
        model and one loaded cross-encoder across every user's `Pipeline`
        instead of paying their load cost - and their RAM - once per user. Every
        single-tenant caller (`serve.py`, `ask.py`, the eval scripts) omits both
        and gets the original one-model-per-instance behaviour unchanged.

        `llm`, likewise, lets the caller inject a `Synthesizer`'s model - e.g. one
        built from a specific user's own pasted API key - instead of the default
        that reads a single `.env` key for the whole process.

        Contract: a caller passing `model` must have loaded it for this
        `index_dir`'s own `model_id` (recorded in `config.json`), and a caller
        passing `reranker` must pass a matching `rerank` spec string alongside it
        - same as the implicit contract these already have with each other in
        the single-model-per-instance case.

        `bulk_sample`, same opt-in shape again: an optional path to the web
        app's per-user `bulk_messages.parquet` (webapp/app/ingestion/worker.py),
        which records only `dedup_key`/`date_utc`/`sender` for mail
        `corpus.filters.is_bulk` drops at ingestion - never subject/body, since
        the filter's whole point is that this mail stays out of search. `None`
        (every CLI/eval caller) means `_message_date_arm` reports only the
        non-bulk count, exactly as before this existed.
        """
        self.index_dir = Path(index_dir)
        self.sample = Path(sample)
        self.bulk_sample = Path(bulk_sample) if bulk_sample is not None else None
        self.retriever = retriever
        self.rerank = rerank
        self.transform = transform
        self.dense_weight = dense_weight
        self.verbose = verbose

        meta_path = self.index_dir / "config.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no built index at {self.index_dir}.\n"
                f"  build one: make index-pilot")
        self.meta = json.loads(meta_path.read_text())
        self.chunking = self.meta["chunking"]
        self.model_id = self.meta["model"]

        E.configure_threads(threads)
        self._log(f"loading {self.index_dir.name} ...")
        self.bm25 = SparseIndex.load(self.index_dir / "bm25")
        self.dense = DenseIndex.load(self.index_dir / "dense")
        self.model = model if model is not None else E.load_model(self.model_id)
        self.instruction = E.QUERY_INSTRUCTION.get(self.model_id, "")

        self.chunk_meta = CT.load_chunk_meta(self.index_dir / "chunks.jsonl")
        self.texts = CT.load_or_build(self.index_dir, self.sample, self.chunking,
                                      verbose=verbose)
        self._messages = self._load_message_metadata()
        self._bulk_messages = self._load_bulk_metadata()

        spec = RR.SPECS.get(rerank)
        self.reranker = reranker
        if self.reranker is None and spec is not None:
            self._log(f"reranker: {spec.label}")
            self.reranker = RR.CrossEncoderReranker.from_spec(spec)
        self.rerank_top_k = spec.top_k if spec else 0

        self.transformer = (QueryTransformer(transform) if transform != "none"
                            else None)
        self.synthesizer = Synthesizer(llm=llm)

        # The router is opt-in because its SQL arm needs extracted commitments,
        # and extraction is an overnight job that has not necessarily been run.
        # With no commitments the SQL arm returns nothing and says so, rather than
        # reporting "nothing is due" - which would be indistinguishable from a
        # correct empty answer.
        self.router = None
        self.commitments: list = []
        if route:
            from .router.classify import QueryRouter
            # Same `llm` this Pipeline was given for the synthesizer, not a
            # second one built from whatever `LLM()`'s own default provider
            # lookup finds - in the multi-tenant web app that would silently
            # use a *different* key than the one answering the question (or
            # none at all, landing on the router's own default-to-"both"
            # exception path rather than a real classification). A caller
            # that swaps `llm=` per request (see chat/routes.py) should swap
            # `self.router._llm` the same way for the same reason.
            self.router = QueryRouter(llm=llm, use_llm=True)
            self.commitments = self._load_commitments(commitments)
            self._log(f"router: on, {len(self.commitments):,} commitments loaded")

        # First encode allocates buffers and materialises lazily-loaded weights -
        # ~1.3 s measured. Paying it at load time keeps the first user query from
        # looking 40x slower than every later one.
        E.encode_queries(self.model, ["warm up"], self.instruction or None)
        self.bm25.search("warm up", top_k=1)
        self._log(f"ready: {len(self.chunk_meta):,} chunks, "
                  f"{len(self._messages):,} messages")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}", flush=True)

    def _load_message_metadata(self) -> dict[str, dict]:
        """dedup_key -> the header fields a citation needs.

        Only headers, never bodies: the body a citation should show is the chunk
        that was retrieved, which is already in `self.texts`. Loading bodies here
        would double the memory for text nobody displays.

        `"date_utc"` keeps the full timestamp (not just `"date"`'s display
        string) for `_message_date_arm` - it has to convert into whichever
        timezone `as_of`/`tz` were resolved in before comparing against a
        window, and a `"%Y-%m-%d"` string has already thrown that conversion
        away. Kept as tz-aware UTC, matching MESSAGE_SCHEMA's own
        `timestamp("us", tz="UTC")` column.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(self.sample, columns=[
            "dedup_key", "sender", "recipients", "subject", "date_utc"])
        out = {}
        dates = []
        for row in table.to_pylist():
            date = row.get("date_utc")
            if date is not None:
                dates.append(date)
            out[row["dedup_key"]] = {
                "sender": row.get("sender") or "",
                "recipients": row.get("recipients") or "",
                "subject": row.get("subject") or "",
                "date": date.strftime("%Y-%m-%d") if date is not None else "",
                "date_utc": date,
            }
        self._corpus_end = _corpus_end_date(dates)
        return out

    def _load_bulk_metadata(self) -> list[dict]:
        """`dedup_key`/`date_utc`/`sender` for mail dropped as bulk at
        ingestion - `[]` when `bulk_sample` was never given (every CLI/eval
        caller), so `_message_date_arm` silently has nothing to add and
        behaves exactly as it did before this existed.

        Loaded eagerly, same reasoning as `_load_message_metadata`: this file
        holds header-scale rows only, never chunk-scale text, so there is no
        memory pressure that would justify deferring it.
        """
        if self.bulk_sample is None or not self.bulk_sample.exists():
            return []
        import pyarrow.parquet as pq

        return pq.read_table(self.bulk_sample).to_pylist()

    # -- retrieval ---------------------------------------------------------

    def _rank(self, texts_dense: list[str], texts_sparse: list[str]
              ) -> list[tuple[str, float]]:
        dense_run: list[tuple[str, float]] = []
        sparse_run: list[tuple[str, float]] = []

        if self.retriever != "bm25":
            qvecs = E.encode_queries(self.model, texts_dense, self.instruction or None)
            runs = [self.dense.search(v, top_k=RETRIEVE_DEPTH) for v in qvecs]
            dense_run = (runs[0] if len(runs) == 1
                         else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))
        if self.retriever != "dense":
            runs = [self.bm25.search(t, top_k=RETRIEVE_DEPTH) for t in texts_sparse]
            sparse_run = (runs[0] if len(runs) == 1
                          else reciprocal_rank_fusion(runs, top_k=RETRIEVE_DEPTH))

        if self.retriever == "bm25":
            return sparse_run
        if self.retriever == "dense":
            return dense_run
        if self.retriever == "hybrid_rrf":
            return reciprocal_rank_fusion([dense_run, sparse_run], top_k=RETRIEVE_DEPTH)
        if self.retriever == "hybrid_weighted":
            return weighted_score_fusion(
                [dense_run, sparse_run],
                [self.dense_weight, 1.0 - self.dense_weight], top_k=RETRIEVE_DEPTH)
        raise ValueError(f"unknown retriever {self.retriever!r}")

    def _load_commitments(self, path) -> list:
        """Extracted commitments from JSONL, for the router's SQL arm.

        JSONL rather than Postgres because the database is optional in this project
        and the router must be measurable without one; `router/sql.py` keeps the
        in-memory predicate identical to the SQL so the two cannot drift.

        A caller that already has `Commitment` rows in hand - the multi-tenant
        web app's per-user Postgres query (`webapp/app/commitments.py`), not a
        JSONL file at all - passes them as a plain `list` here directly rather
        than round-tripping through a file this project's single-user callers
        don't otherwise need.
        """
        if isinstance(path, list):
            return path

        from datetime import date as _date

        from .extraction.schema import Commitment

        root = Path(path) if path else Path("data/commitments")
        files = ([root] if root.is_file()
                 else sorted(root.glob("commitments_*.jsonl")) if root.exists() else [])
        out: list = []
        for file in files:
            for line in file.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("due_at", "due_alternative"):
                    if row.get(key):
                        row[key] = _date.fromisoformat(row[key])
                out.append(Commitment(**row))
        return out

    def _structured_messages(self, commitment_rows: list[dict],
                             date_rows: list[dict]) -> list[RetrievedMessage]:
        """Both date arms' hits, as `RetrievedMessage`s `ask()` already knows
        how to cite - see that method's own citation-building loop, which
        this required no change to. Deduped on message id: a message that
        both has an extracted commitment *and* falls in the date window
        should be cited once, not twice.
        """
        out = [_commitment_row_to_message(r, i, self._messages)
              for i, r in enumerate(commitment_rows)]
        seen = {m.message_id for m in out}
        for row in date_rows:
            if row["dedup_key"] in seen:
                continue
            out.append(_message_date_row_to_message(row, len(out)))
            seen.add(row["dedup_key"])
        return out

    def search(self, question: str, top_n: int = 10, as_of: str = "", tz=None) -> SearchResult:
        """Route if configured, then retrieve, rerank, and collapse to messages.

        `tz` is only used by `_message_date_arm`, and only matters if it was
        resolved in the same zone as `as_of` - a caller passing one without
        the other has a window and a message-date comparison disagreeing
        about what day "today" means. See `filter_messages_by_date`'s
        docstring for why that gap is real, not theoretical.
        """
        timings: dict[str, float] = {}
        route_info: dict | None = None
        commitment_rows: list[dict] = []
        date_rows: list[dict] = []

        if self.router is not None:
            t0 = time.perf_counter()
            decision = self.router.route(question)
            route_info = {
                "route": decision.route, "reason": decision.reason,
                "decided_by": decision.decided_by,
                "confidence": decision.confidence,
                "n_commitments_indexed": len(self.commitments),
            }
            window_resolved = True
            if decision.uses_sql:
                commitment_rows, note = self._sql_arm(question, as_of)
                route_info["sql_note"] = note
                # "What's due" and "what came in" are both date questions;
                # `uses_sql` covers either phrasing, so both arms run and
                # whichever one actually has hits is the one that answers.
                date_rows, date_note = self._message_date_arm(question, as_of, tz=tz)
                route_info["date_note"] = date_note
                # `_TEMPORAL` and `parse_window` are two separate phrase
                # lists that happen to overlap, not one shared source of
                # truth - "what are my deadlines" and "how soon is this due"
                # match `_TEMPORAL` (bare "deadline"/"how soon", no
                # week/today/explicit-date) and rules confidently call them
                # "sql", but `parse_window` has no window phrase to resolve
                # for either one. Same failure reachable via the LLM
                # classifier: observed live, "what's most urgent" routed to
                # "sql" ("implies sorting by due date") with nothing to
                # filter on either. Checked directly rather than sniffed out
                # of `note` - `_sql_arm`'s note says something else entirely
                # ("no commitments have been extracted yet") when
                # `self.commitments` is empty, a check `_message_date_arm`
                # never makes, so the two arms' notes are not
                # interchangeable text for the same condition. A pure-SQL
                # answer with no window to filter on is not a real answer,
                # it is a guaranteed empty one - falling through to
                # retrieval is the same safety net `classify.py` already
                # applies to rule abstention: uncertain is not a reason to
                # guess, and a wrong route is worse than a slow answer.
                from .router.sql import parse_window
                anchor = as_of or self._corpus_end
                window_resolved = parse_window(question, anchor) is not None
            timings["route_ms"] = (time.perf_counter() - t0) * 1000

            if not decision.uses_retrieval and window_resolved:
                # A pure SQL question skips *semantic* retrieval - that is the
                # point of the router: "what's due next week" is a date
                # comparison, and running retrieval for it would only add
                # latency and distractors. It does not skip citing what the
                # date arms above already found.
                structured = self._structured_messages(commitment_rows, date_rows)
                timings.setdefault("transform_ms", 0.0)
                timings["retrieval_ms"] = 0.0
                timings["rerank_ms"] = 0.0
                timings["total_ms"] = sum(timings.values())
                return SearchResult(question=question, messages=structured, timings=timings,
                                    route=route_info, commitments=commitment_rows)

        t0 = time.perf_counter()
        if self.transformer is not None:
            tq = self.transformer(question, as_of=as_of)
        else:
            from .query.transform import identity
            tq = identity(question)
        timings["transform_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ranked = self._rank(tq.dense_texts, tq.sparse_texts)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        timings["rerank_ms"] = 0.0
        if self.reranker is not None:
            # Reranked against the user's actual question, never the transform's
            # output: a HyDE document is a fabricated email, and scoring passages
            # for similarity to a fabrication measures agreement with the
            # generator rather than relevance to the question.
            ranked, ms = self.reranker.rerank_timed(
                question, ranked, self.texts.texts, self.rerank_top_k)
            timings["rerank_ms"] = ms

        messages = self._collapse(ranked, top_n)
        if commitment_rows or date_rows:
            # "Both" route: the date arms' hits are exact matches for a
            # question that named a window, so they lead; semantic results
            # fill whatever room is left rather than being discarded.
            seen = {m.message_id for m in messages}
            structured = [m for m in self._structured_messages(commitment_rows, date_rows)
                         if m.message_id not in seen]
            messages = (structured + messages)[:top_n]
        timings["total_ms"] = sum(timings.values())

        return SearchResult(
            question=question, messages=messages, timings=timings,
            transform_kind=tq.kind,
            transform_texts=[t for t in tq.dense_texts if t != question],
            transform_degraded=tq.degraded,
            route=route_info, commitments=commitment_rows,
        )

    def _sql_arm(self, question: str, as_of: str) -> tuple[list[dict], str]:
        """The router's date-comparison arm. Returns (rows, note).

        The note is not decoration. Three different situations produce an empty
        list - no commitments extracted, no window in the question, an empty window
        - and a UI that showed all three as "nothing is due" would be lying in two
        of them.
        """
        from .router.sql import filter_commitments, parse_window

        if not self.commitments:
            return [], ("no commitments have been extracted yet - run "
                        "scripts/extract_commitments.py. This is not the same as "
                        "'nothing is due'.")

        anchor = as_of or self._corpus_end
        window = parse_window(question, anchor)
        if window is None:
            return [], (f"the question names no time window (anchored on {anchor}), "
                        f"so there is nothing to filter on")

        hits = filter_commitments(self.commitments, window)
        note = f"{len(hits)} commitment(s) due {window.describe()}"
        if not hits:
            note += " - the window is genuinely empty"
        return [c.as_row() for c in hits[:50]], note

    def _message_date_arm(self, question: str, as_of: str, tz=None) -> tuple[list[dict], str]:
        """The router's other date arm: messages filtered by when they
        *arrived*, not a commitment's due date. Runs alongside `_sql_arm`
        whenever a window resolves - "what's due" and "what came in" are
        both date questions, and `_sql_arm` cannot answer the second one no
        matter how it's tuned, since it only ever sees messages extraction
        found an obligation in. See `filter_messages_by_date`'s own
        docstring for why this needed no new index or database - and for
        why `tz` has to be the same zone `as_of` was resolved in.
        """
        from .router.sql import filter_messages_by_date, parse_window

        anchor = as_of or self._corpus_end
        window = parse_window(question, anchor)
        if window is None:
            return [], (f"the question names no time window (anchored on {anchor}), "
                        f"so there is nothing to filter on")

        hits = filter_messages_by_date(self._messages, window, tz=tz)
        bulk_count = self._count_bulk_in_window(window, tz=tz)
        # No new classification pass: "important" reuses the commitment
        # extraction that already ran for the router's SQL arm (a message
        # either has an extracted deadline/obligation or it doesn't), and
        # "general" is simply the non-bulk remainder. Three tiers for free
        # from two signals this system already computes.
        commitment_keys = {c.message_id for c in self.commitments}
        important = sum(1 for h in hits if h.get("dedup_key") in commitment_keys)
        general = len(hits) - important
        # "message(s)" undersells this to anyone who does not already know
        # corpus/filters.py's own bulk-mail rule ran at ingestion: newsletters,
        # no-reply senders and anything with a List-Unsubscribe header never
        # reached this index at all. Silently reporting "3 received" to a
        # person whose real inbox got 20 that day reads as the count being
        # wrong, when what actually happened is 17 were bulk mail this system
        # was built to filter out before it ever counts or cites anything.
        # `bulk_count` is only ever nonzero when `bulk_sample` was given (see
        # `_load_bulk_metadata`) - every CLI/eval caller gets the original
        # wording unchanged, since `self._bulk_messages` is `[]` for them.
        if self.bulk_sample is not None:
            note = (f"{len(hits)} non-bulk message(s) received {window.describe()} "
                   f"- {important} with an extracted deadline/commitment (important), "
                   f"{general} general information - plus {bulk_count} "
                   f"promotional/newsletter message(s) filtered from the index and "
                   f"not among the excerpts below ({len(hits) + bulk_count} total "
                   f"received)")
        else:
            note = (f"{len(hits)} non-bulk message(s) received {window.describe()} - "
                   f"promotional/newsletter mail is dropped at ingestion and not "
                   f"counted here")
        if not hits:
            note += "; the window is genuinely empty"
        return hits[:50], note

    def _count_bulk_in_window(self, window, tz=None) -> int:
        """How many bulk-filtered messages (see `_load_bulk_metadata`) arrived
        in `window` - the other half of `_message_date_arm`'s honest total.
        Mirrors `filter_messages_by_date`'s own tz-conversion reasoning
        exactly, just counting rather than collecting rows: there is nothing
        here for a citation to point at, since bulk rows carry no
        subject/body to show.
        """
        from datetime import timezone as _timezone

        zone = tz or _timezone.utc
        count = 0
        for row in self._bulk_messages:
            raw = row.get("date_utc")
            if raw is None:
                continue
            if window.start <= raw.astimezone(zone).date() <= window.end:
                count += 1
        return count

    def _collapse(self, ranked: list[tuple[str, float]], top_n: int
                  ) -> list[RetrievedMessage]:
        return collapse(ranked, top_n, self.texts.texts, self._messages)

    # -- generation --------------------------------------------------------

    def ask(self, question: str, n_sources: int = 6, as_of: str = "",
            top_n: int = 10, tz=None, mailbox_owner: str = "",
            include_route_notes: bool = False) -> tuple[Answer, SearchResult]:
        """The whole thing: retrieve, rerank, synthesize a cited answer.

        `mailbox_owner` only reaches the synthesizer's prompt, never
        `search()` - see `Synthesizer.answer`'s own docstring for why it
        exists. Folding it into `question` instead would also feed it to
        retrieval, diluting the embedding/BM25 query with boilerplate it
        gains nothing from.

        `include_route_notes` is the same kind of opt-in, for the same reason:
        off by default so the CLI/eval harness's prompt is untouched, on for
        the web app. When a date/SQL arm ran, `_sql_arm`/`_message_date_arm`
        already computed the *true* count before `n_sources` truncates
        `citations` to a handful of examples - without passing that count
        through, "how many" is answered by counting whatever sample survived
        truncation, which undercounts the moment the real total exceeds
        `n_sources`. The eval harness never asks "how many", so this only
        matters for the web app's live chat.
        """
        result = self.search(question, top_n=max(top_n, n_sources), as_of=as_of, tz=tz)
        citations = [
            Citation(n=i + 1, message_id=m.message_id, sender=m.sender,
                     recipients=m.recipients, date=m.date, subject=m.subject,
                     text=m.text, score=m.score, chunk_ids=list(m.chunk_ids))
            for i, m in enumerate(result.messages[:n_sources])
        ]
        route_note = ""
        if include_route_notes and result.route:
            notes = [n for n in (result.route.get("sql_note"),
                                 result.route.get("date_note")) if n]
            route_note = "; ".join(notes)
        self.synthesizer.n_sources = n_sources
        answer = self.synthesizer.answer(question, citations, as_of=as_of,
                                         mailbox_owner=mailbox_owner,
                                         route_note=route_note)
        return answer, result

    @property
    def config_summary(self) -> dict:
        return {
            "index": self.index_dir.name,
            "chunking": self.chunking,
            "model": self.model_id,
            "retriever": self.retriever,
            "rerank": self.rerank,
            "transform": self.transform,
            "n_chunks": len(self.chunk_meta),
            "n_messages": len(self._messages),
            "embed_platform": self.meta.get("embed_platform", "local-cpu"),
        }
