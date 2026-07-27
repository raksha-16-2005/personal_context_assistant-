#!/usr/bin/env python
"""Hand-verify candidate queries into the published eval set.

Uses **pooling**, the standard TREC construction: for each query, take the
union of what several retrievers return, and have a human judge every message
in that pool. Judging all 50k messages per query is obviously impossible;
judging only the source thread would label a query relevant to exactly the
thread it was drafted from and silently mark every other genuine answer
non-relevant, which punishes any retriever good enough to find them.

The pool here is BM25 top-k, dense top-k, and the full source thread. The
retrievers disagree substantially, which is what makes the union worth more
than either alone.

**The known bias, which belongs in EVALUATION.md:** a relevant message that no
pooled retriever surfaced is never judged, so it counts as non-relevant.
Absolute recall is therefore slightly optimistic for every system. It is
*comparably* optimistic, so between-config comparisons - which is all the
ablation tables claim - stay sound. Widening `--pool-k` shrinks the bias at
the cost of your labelling time.

Judgements save after every query. Losing three hours of labelling to a
stray Ctrl-C is not a risk worth taking.

    python scripts/verify_eval.py --index data/index/thread_aware__all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.index import embed as E  # noqa: E402
from emailrag.index.dense import DenseIndex  # noqa: E402
from emailrag.index.sparse import SparseIndex  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, CYAN, YELLOW = "\033[32m", "\033[31m", "\033[36m", "\033[33m"


def getkey() -> str:
    """Read one keypress without waiting for Enter.

    Falls back to line input when stdin is not a tty, so the script still
    works under a pipe or in CI.
    """
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "n")[:1]
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x03":            # Ctrl-C in raw mode does not raise
        raise KeyboardInterrupt
    return ch.lower()


def build_pool(query: str, source_thread: str, bm25: SparseIndex, dense: DenseIndex,
               model, instruction: str, by_key: dict, by_thread: dict,
               pool_k: int) -> list[tuple[str, str]]:
    """Return [(dedup_key, provenance)] - the messages a human must judge."""
    rank: dict[str, list[str]] = {}

    for i, (chunk_id, _s) in enumerate(bm25.search(query, top_k=pool_k * 2), start=1):
        rank.setdefault(chunk_id.rsplit(":", 1)[0], []).append(f"bm25 #{i}")

    qvec = E.encode_queries(model, [query], instruction or None)[0]
    for i, (chunk_id, _s) in enumerate(dense.search(qvec, top_k=pool_k * 2), start=1):
        rank.setdefault(chunk_id.rsplit(":", 1)[0], []).append(f"dense #{i}")

    # The drafting thread is always judged: its messages are the most likely
    # answers and a retriever that misses them should be penalised.
    for msg in by_thread.get(source_thread, []):
        rank.setdefault(msg["dedup_key"], []).append("source thread")

    def best(key: str) -> int:
        ns = [int(p.split("#")[1]) for p in rank[key] if "#" in p]
        return min(ns) if ns else 0    # source-thread-only sorts first

    keys = [k for k in rank if k in by_key]
    keys.sort(key=lambda k: (best(k), k))
    return [(k, ", ".join(rank[k])) for k in keys[:pool_k]]


def show(msg: dict, provenance: str, qi: int, qn: int, pi: int, pn: int,
         query: str, cls: str, n_rel: int, full: bool, as_of: str = "") -> None:
    date = msg["date_utc"].strftime("%Y-%m-%d") if msg["date_utc"] else "unknown"
    body = (msg["body_new"] or "").strip()
    shown = body if full else body[:700]
    truncated = len(body) > len(shown)

    print("\033[2J\033[H", end="")   # clear
    print(f"{DIM}{'─'*78}{RESET}")
    print(f" {BOLD}query {qi}/{qn}{RESET}  {CYAN}[{cls}]{RESET}  {BOLD}{query}{RESET}")
    # Temporal queries are asked as of a fixed date; without it on screen you
    # cannot tell whether a deadline in a message actually satisfies the query.
    if as_of:
        print(f" {YELLOW}asked as of {as_of}{RESET}")
    print(f" {DIM}pool {pi}/{pn}   marked relevant so far: {n_rel}{RESET}")
    print(f"{DIM}{'─'*78}{RESET}")
    print(f" From:    {msg['sender']}")
    print(f" To:      {(msg['recipients'] or '').replace(';', ', ')[:64]}")
    print(f" Date:    {date}")
    print(f" Subject: {msg['subject'][:64]}")
    print(f" {DIM}({provenance}){RESET}")
    print(f"{DIM}{'─'*78}{RESET}")
    print(shown)
    if truncated:
        print(f"{DIM}... [{len(body)-len(shown)} more chars - press m]{RESET}")
    print(f"{DIM}{'─'*78}{RESET}")
    print(f" {GREEN}[r]{RESET}elevant  {RED}[n]{RESET}ot  [m]ore  [b]ack  "
          f"{YELLOW}[s]{RESET}kip query  [q]uit+save")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/eval/candidates.jsonl", type=Path)
    ap.add_argument("--out", default="data/eval/queries.jsonl", type=Path)
    ap.add_argument("--sample", default="data/interim/sample.parquet", type=Path)
    ap.add_argument("--index", required=True, type=Path,
                    help="a built config dir under data/index/")
    ap.add_argument("--pool-k", type=int, default=30,
                    help="messages judged per query; higher = less pooling bias, more work")
    args = ap.parse_args()

    for p in (args.candidates, args.sample, args.index):
        if not p.exists():
            print(f"error: {p} missing", file=sys.stderr)
            return 1

    cfg = json.loads((args.index / "config.json").read_text())
    print(f"index: {cfg['chunking']} / {cfg['model']}  ({cfg['n_chunks']:,} chunks)")

    rows = pq.read_table(args.sample).to_pylist()
    by_key = {r["dedup_key"]: r for r in rows}
    by_thread: dict[str, list[dict]] = {}
    for r in rows:
        by_thread.setdefault(r["thread_id"], []).append(r)

    candidates = [json.loads(l) for l in args.candidates.read_text().splitlines() if l.strip()]

    # Resume: keep finished judgements, re-present only what is unverified.
    done: dict[str, dict] = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                q = json.loads(line)
                if q.get("verified"):
                    done[q["query_id"]] = q
        print(f"resuming: {len(done)} queries already verified")

    todo = [c for c in candidates if c["query_id"] not in done]
    if not todo:
        print("nothing left to verify")
        return 0

    print(f"loading retrievers ...")
    E.configure_threads(6)
    bm25 = SparseIndex.load(args.index / "bm25")
    dense = DenseIndex.load(args.index / "dense")
    model = E.load_model(cfg["model"])
    instruction = E.QUERY_INSTRUCTION.get(cfg["model"], "")

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged = [done[c["query_id"]] if c["query_id"] in done else c for c in candidates]
        with open(args.out, "w") as fh:
            for q in merged:
                fh.write(json.dumps(q) + "\n")

    quit_now = False
    for qi, cand in enumerate(todo, start=1):
        pool = build_pool(cand["query"], cand.get("source_thread_id", ""), bm25, dense,
                          model, instruction, by_key, by_thread, args.pool_k)
        relevant: list[str] = []
        pi, full = 0, False

        while pi < len(pool):
            key, provenance = pool[pi]
            show(by_key[key], provenance, qi, len(todo), pi + 1, len(pool),
                 cand["query"], cand["query_class"], len(relevant), full,
                 as_of=cand.get("as_of", ""))
            try:
                k = getkey()
            except KeyboardInterrupt:
                quit_now = True
                break

            if k == "r":
                if key not in relevant:
                    relevant.append(key)
                pi, full = pi + 1, False
            elif k == "n":
                if key in relevant:
                    relevant.remove(key)
                pi, full = pi + 1, False
            elif k == "m":
                full = True
            elif k == "b":
                pi, full = max(0, pi - 1), False
            elif k == "s":
                break
            elif k == "q":
                quit_now = True
                break

        if quit_now:
            break

        cand["relevant_message_ids"] = relevant
        cand["verified"] = True
        done[cand["query_id"]] = cand
        save()                          # after every query, not at the end

    save()
    n_done = sum(1 for c in candidates if done.get(c["query_id"], {}).get("verified"))
    print(f"\n{n_done}/{len(candidates)} verified -> {args.out}")
    print(f"validate with: .venv/bin/python scripts/validate_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
