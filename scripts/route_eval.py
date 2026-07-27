#!/usr/bin/env python
"""Router accuracy, and end-to-end quality per query class.

    python scripts/route_eval.py                  # router accuracy table
    python scripts/route_eval.py --no-llm         # rules only, no quota spent

The per-class table is the project's headline result, because the claim it tests is
a claim about a failure: dense retrieval collapses on temporal questions, and a
corpus-wide average hides that completely. Router accuracy is reported beside it,
decomposed into what the rules decided and what the model decided - a single
end-to-end number could not distinguish "the router works" from "the regexes
happened to cover the eval set".

`both` counts as correct when it includes the right arm. The router's job is not
to be minimal, it is to not miss; the over-routed column is what that policy costs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation.evalset import load as load_eval  # noqa: E402
from emailrag.router.classify import (  # noqa: E402
    CLASS_TO_ROUTE,
    QueryRouter,
    routes_table,
    score_routes,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--no-llm", action="store_true",
                    help="rules only - abstentions default to `both`")
    ap.add_argument("--list", action="store_true", help="print every decision")
    args = ap.parse_args()

    if not args.eval.exists():
        print(f"error: {args.eval} missing - label some queries first:\n"
              f"  make verify", file=sys.stderr)
        return 1

    queries = [q for q in load_eval(args.eval) if q.verified]
    if not queries:
        print("error: no verified queries in the eval set", file=sys.stderr)
        return 1

    router = QueryRouter(use_llm=not args.no_llm)
    decisions = [router.route(q.query) for q in queries]
    classes = {q.query: q.query_class for q in queries}

    score = score_routes(decisions, classes)
    print(f"{len(queries)} verified queries, "
          f"{score['n']} scoreable (unanswerable controls excluded - there is no "
          f"right arm for a question with no answer)\n")
    print(routes_table(score))
    print(f"\ndecided by: {router.render_counts()}")
    print(f"llm calls: {sum(d.llm_calls for d in decisions)}")

    if args.list:
        print(f"\n{'-'*70}")
        for q, d in zip(queries, decisions):
            want = CLASS_TO_ROUTE.get(q.query_class)
            verdict = ("n/a" if want is None else
                       "ok" if d.route in (want, "both") else "WRONG")
            over = " (over-routed)" if d.route == "both" and want not in (None, "both") else ""
            print(f"  [{verdict:5s}] {q.query_class:12s} -> {d.route:6s}{over}  "
                  f"{d.query[:60]}")
            print(f"           {d.decided_by}: {d.reason[:80]}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"router_{stamp}.json"
    path.write_text(json.dumps({
        "generated_utc": stamp,
        "n_queries": len(queries),
        "use_llm": not args.no_llm,
        "decided_by": router.counts,
        "score": score,
        "decisions": [{"query": d.query, "class": classes.get(d.query),
                       "route": d.route, "decided_by": d.decided_by,
                       "reason": d.reason} for d in decisions],
    }, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
