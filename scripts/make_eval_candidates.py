#!/usr/bin/env python
"""Draft candidate eval queries from real threads.

Candidates only. Nothing this script emits is a label - every query still has
to go through `verify_eval.py`, where a human decides which messages actually
answer it. An LLM-generated query paired with LLM-guessed relevance labels
would make the whole eval set a measurement of the drafting model, and every
downstream number would inherit its blind spots.

Thread selection is deliberate, not random:

- **semantic / entity** come from multi-message threads with real back-and-forth.
  A singleton thread rarely contains a decision worth asking about.
- **temporal** comes from threads whose text carries date language, since a
  temporal query needs a real deadline behind it to be answerable at all.
- **unanswerable** controls are drafted *about* a sampled thread's topic but
  asking something the corpus cannot answer. Generating them from thin air
  produces queries so off-topic that any retriever refuses trivially, which
  makes the refusal metric meaningless.

    export GEMINI_API_KEY=...
    python scripts/make_eval_candidates.py --n-per-class 5    # pilot
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.llm.client import LLM, LLMError, MissingKey, QuotaExhausted  # noqa: E402

DATE_LANGUAGE = re.compile(
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|\btomorrow\b|\bnext week\b"
    r"|\bby (?:the )?\d{1,2}(?:st|nd|rd|th)?\b|\bdeadline\b|\bdue\b|\beod\b"
    r"|\bcob\b|\bby friday\b|\bthis week\b|\bend of (?:the )?(?:week|month|day)\b",
    re.IGNORECASE,
)

SYSTEM = (
    "You write evaluation queries for an email retrieval benchmark. "
    "You write the kind of question a person would actually type into a search box "
    "over their own mailbox - short, natural, no corporate padding. "
    "Return ONLY valid JSON. No commentary, no markdown fences."
)

CLASS_PROMPTS = {
    "semantic": """Below is one email thread.

Write {n} search queries a participant might later type to find this thread,
asking about its TOPIC, DECISION, or CONTENT.

Rules:
- Natural phrasing, 4-12 words. Lowercase is fine.
- Do NOT quote distinctive exact phrases from the thread. A query that copies a
  rare literal string is trivially solved by BM25 and measures nothing.
- Ask about substance: what was decided, what the disagreement was, what the
  status is.

Return: {{"queries": ["...", "..."]}}

THREAD:
{thread}""",

    "temporal": """Below is one email thread that mentions dates, deadlines, or scheduling.
Today's date, for the purposes of these queries, is {as_of}.

Write {n} search queries about TIME: what is due when, what happens in a given
period, how many deadlines fall in a span.

Rules:
- Natural phrasing, 4-12 words.
- The query MUST depend on resolving a date - that is the whole point of this
  class.
- Anchor every query to a date that is actually recoverable. Either name it
  outright ("what was due the week of {as_of}") or phrase it relative to
  {as_of} ("what is due the following Thursday").
- Do NOT write floating relative queries like "what happened last Sunday" or
  "what is due next week" with no stated anchor. Over a fixed historical
  archive those resolve to nothing and are unanswerable by construction.
- Mix specific ("what was due the week of {as_of}") and aggregate
  ("how many deadlines fell that month").

Return: {{"queries": ["...", "..."]}}

THREAD:
{thread}""",

    "entity": """Below is one email thread.

Write {n} search queries scoped to a specific PERSON or ORGANISATION in it,
combined with a topic. Example shape: "everything from <person> about <topic>".

Rules:
- Use names that actually appear in the thread.
- Natural phrasing, 4-12 words.

Return: {{"queries": ["...", "..."]}}

THREAD:
{thread}""",

    "unanswerable": """Below is one email thread.

Write {n} search queries that are PLAUSIBLE for this mailbox and closely related
to the thread's subject matter, but that the thread CANNOT answer.

Rules:
- Same domain and vocabulary as the thread - these must look answerable.
- The answer must genuinely not be present. Ask about a detail that was never
  discussed, a figure never stated, an outcome never reported.
- Do NOT ask about the future or anything time-dependent.
- Natural phrasing, 4-12 words.

Return: {{"queries": ["...", "..."]}}

THREAD:
{thread}""",
}


def render_thread(messages: list[dict], max_chars: int = 2200,
                  per_message_chars: int = 500, max_messages: int = 6) -> str:
    """Render a thread compactly enough to draft queries from.

    Kept deliberately tight. Gemini's free tier meters **input** tokens, and
    sending full 6,000-character threads exhausted a day's quota in about five
    calls. Drafting a search query needs the gist of a thread - who, about
    what, roughly when - not its full text, and the human verifying the query
    reads the real messages anyway.
    """
    ordered = sorted(messages, key=lambda x: (x["date_utc"] is None, x["date_utc"]))
    if len(ordered) > max_messages:
        # Keep the opening and the tail: the request and its resolution carry
        # the thread's substance, the middle is usually logistics.
        ordered = ordered[:2] + ordered[-(max_messages - 2):]

    parts = []
    for m in ordered:
        date = m["date_utc"].strftime("%Y-%m-%d") if m["date_utc"] else "unknown"
        to = (m["recipients"] or "").replace(";", ", ")[:80]
        body = " ".join((m["body_new"] or "").split())[:per_message_chars]
        parts.append(f"[{date}] {m['sender']} -> {to}\n"
                     f"Subject: {m['subject'][:100]}\n{body}")
    return ("\n\n---\n\n".join(parts))[:max_chars]


def thread_as_of(messages: list[dict]) -> str:
    """The date a temporal query about this thread is asked 'as of'."""
    dates = [m["date_utc"] for m in messages if m["date_utc"] is not None]
    return max(dates).strftime("%Y-%m-%d") if dates else "2001-06-01"


def pick_threads(by_thread: dict[str, list[dict]], cls: str, n: int,
                 rng: random.Random, used: set[str]) -> list[str]:
    """Choose source threads suited to the query class."""
    def substantial(rows: list[dict]) -> bool:
        return len(rows) >= 2 and sum(len(r["body_new"] or "") for r in rows) > 600

    if cls == "temporal":
        pool = [t for t, rows in by_thread.items()
                if t not in used and substantial(rows)
                and any(DATE_LANGUAGE.search(r["body_new"] or "") for r in rows)]
    else:
        pool = [t for t, rows in by_thread.items() if t not in used and substantial(rows)]

    pool.sort()                 # determinism before shuffling
    rng.shuffle(pool)
    return pool[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--out", default="data/eval/candidates.jsonl", type=Path)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-per-class", type=int, default=5,
                    help="threads per class; each yields ~2 queries")
    ap.add_argument("--queries-per-thread", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--classes", nargs="*", default=list(CLASS_PROMPTS))
    args = ap.parse_args()

    if not args.sample.exists():
        print(f"error: {args.sample} missing - run `make sample` first", file=sys.stderr)
        return 1

    try:
        llm = LLM(args.provider, args.model, temperature=0.7)  # variety, not determinism
    except MissingKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = pq.read_table(args.sample).to_pylist()
    by_thread: dict[str, list[dict]] = {}
    for r in rows:
        by_thread.setdefault(r["thread_id"], []).append(r)
    print(f"{len(rows):,} messages in {len(by_thread):,} threads")

    rng = random.Random(args.seed)
    used: set[str] = set()
    out: list[dict] = []
    n = 0

    out_of_quota = False
    for cls in args.classes:
        if out_of_quota:
            break
        threads = pick_threads(by_thread, cls, args.n_per_class, rng, used)
        print(f"\n{cls}: {len(threads)} source threads")
        if not threads:
            print(f"  warning: no thread matched the criteria for {cls}")
        for tid in threads:
            used.add(tid)
            # Anchor temporal queries to the last message in the source thread:
            # that is the moment a participant would plausibly have asked, and
            # it makes relative phrasing ("the following Thursday") resolvable.
            as_of = thread_as_of(by_thread[tid])
            prompt = CLASS_PROMPTS[cls].format(
                n=args.queries_per_thread, as_of=as_of,
                thread=render_thread(by_thread[tid]))
            try:
                data = llm.json_complete(prompt, SYSTEM, max_tokens=800)
                queries = data["queries"] if isinstance(data, dict) else data
            except QuotaExhausted as exc:
                # Every model is spent - the next class would fail identically.
                # Stop the whole run and keep what we have; partial candidates
                # are still worth verifying, and re-running resumes tomorrow.
                print(f"\n{exc}")
                out_of_quota = True
                break
            except (LLMError, KeyError, TypeError) as exc:
                print(f"  {tid[:24]}: FAILED {str(exc)[:120]}")
                continue
            for q in queries:
                if not isinstance(q, str) or not q.strip():
                    continue
                n += 1
                out.append({
                    "query_id": f"q{n:03d}",
                    "query": q.strip(),
                    "query_class": cls,
                    "relevant_message_ids": [],
                    "verified": False,
                    "source_thread_id": tid,
                    "as_of": as_of if cls == "temporal" else "",
                    "notes": "",
                })
                print(f"  {q.strip()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")

    got = {cls: sum(1 for r in out if r["query_class"] == cls) for cls in args.classes}
    print(f"\n{len(out)} candidates -> {args.out}")
    print("  " + "  ".join(f"{c}={n}" for c, n in got.items()))
    if out_of_quota:
        print("\nstopped early on quota. Gemini's free tier resets daily - "
              "re-run tomorrow to top up the thin classes, or label these now.")
    print(f"next: make verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
