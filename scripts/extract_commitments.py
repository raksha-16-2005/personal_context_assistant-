#!/usr/bin/env python
"""Run commitment extraction over the corpus.

    # local arm, scoped to what the temporal and entity queries touch
    python scripts/extract_commitments.py --scope eval --provider ollama

    # hosted ceiling over the same messages, for the comparison
    python scripts/extract_commitments.py --scope eval --provider anthropic

    # compare the two arms
    python scripts/extract_commitments.py --compare

Scope matters more than it looks. Ollama is CPU-only on this machine at ~25 s per
message, so the full 50k sample is ~350 hours - not a run, a season. `--scope eval`
restricts extraction to the threads the temporal and entity queries actually touch,
which is all the router needs to be measured and is one overnight job. The
pre-filter (see extraction/dates.py) then drops most of even that.

Output is JSONL rather than straight to Postgres, deliberately: an overnight run
should survive the database not being up, and re-resolving dates under a changed
convention should not need a re-extraction at 25 s/message.
`scripts/load_commitments.py` moves JSONL into the table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.extraction.extract import CommitmentExtractor  # noqa: E402
from emailrag.extraction.metrics import compare_arms  # noqa: E402
from emailrag.extraction.schema import Commitment  # noqa: E402
from emailrag.llm.client import LLM  # noqa: E402

COLUMNS = ["dedup_key", "thread_id", "sender", "recipients", "subject",
           "body_new", "date_utc"]

# Extraction arms. The comparison between them is the result: a local model that
# reaches most of the ceiling's date accuracy means private mail never leaves the
# machine, which is the entire argument for phase 7.
ARMS = {
    "ollama": ("qwen2.5:3b", "local, CPU-only, ~25 s/message"),
    "anthropic": ("claude-haiku-4-5-20251001", "hosted ceiling, ~$3 batched"),
    "gemini": (None, "hosted, free tier"),
}


def eval_scoped_messages(messages: list[dict], eval_path: Path) -> list[dict]:
    """Messages in the threads the temporal and entity queries touch.

    Whole threads, not individual labelled messages: a commitment is often stated
    in one message and dated in the reply, and the router is asked about the thread.
    """
    from emailrag.evaluation.evalset import load as load_eval

    if not eval_path.exists():
        print(f"warning: {eval_path} missing - falling back to --scope all",
              file=sys.stderr)
        return messages

    queries = [q for q in load_eval(eval_path)
               if q.query_class in ("temporal", "entity")]
    labelled = {m for q in queries for m in q.relevant_message_ids}
    by_key = {m["dedup_key"]: m for m in messages}
    threads = {by_key[k]["thread_id"] for k in labelled if k in by_key}

    scoped = [m for m in messages if m.get("thread_id") in threads]
    print(f"  scope=eval: {len(queries)} temporal/entity queries -> "
          f"{len(labelled)} labelled messages -> {len(threads)} threads -> "
          f"{len(scoped):,} messages")
    return scoped


def load_jsonl(path: Path) -> list[Commitment]:
    from datetime import date

    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("due_at", "due_alternative"):
            if row.get(key):
                row[key] = date.fromisoformat(row[key])
        out.append(Commitment(**row))
    return out


def do_compare(args) -> int:
    runs = sorted(args.out_dir.glob("commitments_*.jsonl"))
    if len(runs) < 2:
        print(f"error: need two arms in {args.out_dir}; found {len(runs)}.\n"
              f"  run with --provider ollama, then --provider anthropic",
              file=sys.stderr)
        return 1

    by_model: dict[str, list[Commitment]] = {}
    for path in runs:
        for c in load_jsonl(path):
            by_model.setdefault(c.model, []).append(c)
    if len(by_model) < 2:
        print(f"error: all runs are from one model ({list(by_model)})", file=sys.stderr)
        return 1

    models = sorted(by_model, key=lambda m: len(by_model[m]))
    local, ceiling = models[0], models[-1]
    gold = _load_gold(args.gold)

    cmp = compare_arms(by_model[local], by_model[ceiling], gold=gold,
                       local_model=local, ceiling_model=ceiling)
    print(f"\n{cmp.render()}")
    if not gold:
        print("\nNo gold labels, so the date-accuracy rows are empty. Hand-label a "
              f"sample into {args.gold} to fill them - agreement between two models "
              "is not accuracy.")
    return 0


def _load_gold(path: Path) -> dict[str, list[Commitment]]:
    if not path.exists():
        return {}
    gold: dict[str, list[Commitment]] = {}
    for c in load_jsonl(path):
        gold.setdefault(c.message_id, []).append(c)
    print(f"  gold: {sum(len(v) for v in gold.values())} labelled commitments "
          f"over {len(gold)} messages")
    return gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/commitments"))
    ap.add_argument("--gold", type=Path,
                    default=Path("data/commitments/gold.jsonl"),
                    help="hand-labelled commitments, for date accuracy")
    ap.add_argument("--provider", default="ollama", choices=sorted(ARMS))
    ap.add_argument("--model", default=None, help="override the arm's model")
    ap.add_argument("--scope", default="eval", choices=["eval", "all"])
    ap.add_argument("--since", default="",
                    help="YYYY-MM-DD; only messages sent on or after this date. "
                         "Phase 7 caps extraction at the recent window.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N messages")
    ap.add_argument("--no-prefilter", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="compare existing runs instead of extracting")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        return do_compare(args)

    if not args.sample.exists():
        print(f"error: {args.sample} missing - run `make sample`", file=sys.stderr)
        return 1

    messages = pq.read_table(args.sample, columns=COLUMNS).to_pylist()
    print(f"{len(messages):,} messages in the sample")

    if args.scope == "eval":
        messages = eval_scoped_messages(messages, args.eval)
    if args.since:
        cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        before = len(messages)
        messages = [m for m in messages
                    if m.get("date_utc") and m["date_utc"] >= cutoff]
        print(f"  --since {args.since}: {before:,} -> {len(messages):,}")
    if args.limit:
        messages = messages[:args.limit]

    if not messages:
        print("error: no messages in scope", file=sys.stderr)
        return 1

    model = args.model or ARMS[args.provider][0]
    llm = LLM(args.provider, model=model)
    print(f"\narm: {args.provider} / {llm.model}  ({ARMS[args.provider][1]})")
    print(f"messages in scope: {len(messages):,}")
    if args.provider == "ollama":
        est = len(messages) * 25 / 3600
        print(f"projected worst case: {est:.1f} h if nothing is prefiltered or "
              f"cached\n")

    extractor = CommitmentExtractor(llm, prefilter=not args.no_prefilter)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"commitments_{args.provider}_{stamp}.jsonl"

    t0 = time.time()
    written = 0
    # Written as they are produced. An overnight run that dies in hour six must
    # keep the first six hours, and the LLM cache means a restart re-reads rather
    # than re-computes.
    with open(out_path, "w") as fh:
        for i, message in enumerate(messages, start=1):
            for c in extractor.extract(message):
                fh.write(json.dumps(c.as_row()) + "\n")
                written += 1
            if i % 25 == 0 or i == len(messages):
                fh.flush()
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(messages) - i) / rate / 60 if rate else 0
                print(f"  {i:,}/{len(messages):,}  {written} commitments  "
                      f"{rate:.1f} msg/s  eta {eta:.0f} min", flush=True)

    print(f"\n{extractor.stats.render()}")
    print(f"\nwrote {out_path}  ({written} commitments)")

    if extractor.errors:
        err_path = out_path.with_suffix(".errors.json")
        err_path.write_text(json.dumps(extractor.errors[:500], indent=2))
        print(f"      {err_path}  ({len(extractor.errors)} errors)")

    summary = args.out_dir / f"stats_{args.provider}_{stamp}.json"
    summary.write_text(json.dumps({
        "provider": args.provider, "model": llm.model, "scope": args.scope,
        "since": args.since, "prefilter": not args.no_prefilter,
        **{k: getattr(extractor.stats, k) for k in (
            "messages_seen", "messages_prefiltered_out", "messages_called",
            "commitments", "with_due_date", "ambiguous_dates", "rolled_year",
            "unresolvable_phrases", "invalid_rows", "failures", "cached_calls")},
        "prefilter_pass_rate": round(extractor.stats.prefilter_pass_rate, 4),
        "seconds": round(extractor.stats.seconds, 1),
    }, indent=2))
    print(f"      {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
