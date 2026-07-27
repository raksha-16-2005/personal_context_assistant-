#!/usr/bin/env python
"""Ask the corpus a question and get a cited answer.

    python scripts/ask.py "what was decided about the discount tier"
    python scripts/ask.py --search-only "calpine renewal"
    python scripts/ask.py -i                      # interactive

This is the assembled system rather than the benchmark: retrieval, reranking and
generation with citations, on one question, with output meant for a person. It is
also the fastest way to see whether the pipeline is actually working - a metric
can look healthy while the answers are nonsense.

Every LLM call is cached (`data/llm_cache/`), so re-asking the same question costs
nothing. `EMAILRAG_LLM_CACHE=0` forces a live call.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.generation.synthesize import INSUFFICIENT  # noqa: E402
from emailrag.pipeline import DEFAULT_RERANK, Pipeline  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, YELLOW, RED = "\033[36m", "\033[33m", "\033[31m"


def default_index(root: Path) -> Path | None:
    built = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "config.json").exists()) if root.exists() else []
    return built[0] if built else None


def show_sources(messages, limit: int, highlight: set[int]) -> None:
    for i, m in enumerate(messages[:limit], start=1):
        mark = f"{CYAN}[{i}]{RESET}" if i in highlight else f"{DIM}[{i}]{RESET}"
        who = m.sender or "unknown"
        print(f"\n{mark} {BOLD}{m.subject or '(no subject)'}{RESET}")
        print(f"    {DIM}{who} -> {(m.recipients or '')[:60]}  {m.date}  "
              f"score {m.score:.4f}  {len(m.chunk_ids)} chunk(s){RESET}")
        snippet = " ".join(m.text.split())[:280]
        print(f"    {snippet}{'…' if len(m.text) > 280 else ''}")


def run_one(pipe: Pipeline, question: str, args) -> None:
    if args.search_only:
        result = pipe.search(question, top_n=args.top_n, as_of=args.as_of)
        print(f"\n{BOLD}{len(result.messages)} message(s){RESET} "
              f"{DIM}retrieval {result.timings['retrieval_ms']:.0f}ms  "
              f"rerank {result.timings['rerank_ms']:.0f}ms{RESET}")
        show_sources(result.messages, args.top_n, set())
        return

    t0 = time.perf_counter()
    answer, result = pipe.ask(question, n_sources=args.n_sources,
                              as_of=args.as_of, top_n=args.top_n)
    wall = (time.perf_counter() - t0) * 1000

    print()
    if answer.error:
        print(f"{RED}generation failed: {answer.error}{RESET}")
        print(f"{DIM}retrieval still worked - the sources are below.{RESET}")
    elif answer.refused:
        print(f"{YELLOW}No answer in the corpus.{RESET} The model returned "
              f"{INSUFFICIENT}, which is the correct outcome for a question this "
              f"corpus cannot answer.")
    else:
        print(f"{BOLD}{answer.text}{RESET}")

    if answer.invalid_citations:
        print(f"\n{RED}fabricated citation(s): "
              f"{answer.invalid_citations} - only {len(answer.citations)} sources "
              f"were supplied{RESET}")
    if answer.uncited_sentences:
        print(f"\n{YELLOW}uncited claim(s):{RESET}")
        for s in answer.uncited_sentences:
            print(f"  {DIM}- {s[:110]}{RESET}")

    if result.transform_kind != "none" and result.transform_texts:
        print(f"\n{DIM}{result.transform_kind}: "
              f"{'; '.join(t[:70] for t in result.transform_texts[:3])}{RESET}")

    print(f"\n{DIM}{'-'*70}{RESET}")
    show_sources(result.messages, args.top_n, set(answer.cited_numbers))

    t = result.timings
    print(f"\n{DIM}transform {t['transform_ms']:.0f}ms  "
          f"retrieval {t['retrieval_ms']:.0f}ms  rerank {t['rerank_ms']:.0f}ms  "
          f"generation {answer.latency_ms:.0f}ms"
          f"{' (cached)' if answer.cached else ''}  total {wall:.0f}ms{RESET}")
    if answer.model:
        print(f"{DIM}model {answer.model}  cited {answer.cited_numbers}{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question to ask")
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--index-root", type=Path, default=Path("data/index"))
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--retriever", default="hybrid_rrf",
                    choices=["bm25", "dense", "hybrid_rrf", "hybrid_weighted"])
    ap.add_argument("--rerank", default=DEFAULT_RERANK,
                    help=f"rerank arm, or 'none' (default {DEFAULT_RERANK})")
    ap.add_argument("--transform", default="none",
                    choices=["none", "hyde", "multi_query", "decompose"])
    ap.add_argument("--n-sources", type=int, default=6,
                    help="messages given to the generator")
    ap.add_argument("--top-n", type=int, default=8, help="messages to display")
    ap.add_argument("--as-of", default="",
                    help="YYYY-MM-DD to resolve relative dates against")
    ap.add_argument("--search-only", action="store_true",
                    help="retrieve without generating (no LLM call, no key needed)")
    ap.add_argument("-i", "--interactive", action="store_true")
    args = ap.parse_args()

    index = args.index or default_index(args.index_root)
    if index is None:
        print(f"error: no built index under {args.index_root}.\n"
              f"  build one: make index-pilot", file=sys.stderr)
        return 1
    if not args.sample.exists():
        print(f"error: {args.sample} missing - run `make sample`", file=sys.stderr)
        return 1

    question = " ".join(args.question).strip()
    if not question and not args.interactive:
        ap.error("give a question, or -i for interactive")

    pipe = Pipeline(index, args.sample, retriever=args.retriever,
                    rerank=args.rerank, transform=args.transform)
    print(f"{DIM}{pipe.config_summary}{RESET}")

    if question:
        run_one(pipe, question, args)
    if not args.interactive:
        return 0

    print(f"\n{DIM}interactive - blank line or ctrl-d to quit{RESET}")
    while True:
        try:
            q = input(f"\n{BOLD}ask>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        run_one(pipe, q, args)


if __name__ == "__main__":
    raise SystemExit(main())
