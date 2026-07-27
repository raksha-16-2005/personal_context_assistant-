#!/usr/bin/env python
"""Generate answers over the eval set and score them.

    python scripts/eval_generation.py                       # generate + judge
    python scripts/eval_generation.py --generate-only        # answers, no judging
    python scripts/eval_generation.py --kappa data/eval/answer_labels.jsonl

Two passes, deliberately separate. Generation is the system under test; judging is
the instrument. Keeping them apart means the judge can be re-run, swapped, or
recalibrated without regenerating answers - and that the answers a human labels for
calibration are byte-identical to the ones the judge scored.

**Refusal is measured only on the unanswerable controls.** Those ten queries exist
for this metric. A refusal on an answerable query is a different failure and is
counted separately, because folding them together makes both uninterpretable.

**The judge is uncalibrated until `--kappa` has hand labels to compare against.**
The report says so in its own output rather than in a footnote somewhere else. An
LLM judge is an instrument, and an uncalibrated instrument reports precision it has
not earned.

To calibrate: run once, hand-label the emitted `answers_*.jsonl` verdicts into a
labels file (one `{"query_id": ..., "sentence": ..., "verdict": ...}` per line), and
re-run with `--kappa`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emailrag.evaluation.evalset import load as load_eval  # noqa: E402
from emailrag.generation.judge import (  # noqa: E402
    GenerationJudge,
    GenerationReport,
    cohens_kappa,
)
from emailrag.pipeline import DEFAULT_RERANK, Pipeline  # noqa: E402


def default_index(root: Path) -> Path | None:
    built = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "config.json").exists()) if root.exists() else []
    return built[0] if built else None


def generate(pipe: Pipeline, queries: list, n_sources: int) -> list[dict]:
    out = []
    for i, q in enumerate(queries, start=1):
        answer, result = pipe.ask(q.query, n_sources=n_sources, as_of=q.as_of)
        out.append({
            "query_id": q.query_id, "query": q.query, "query_class": q.query_class,
            "as_of": q.as_of,
            "answer": answer.text, "refused": answer.refused,
            "cited": answer.cited_numbers,
            "invalid_citations": answer.invalid_citations,
            "uncited_sentences": answer.uncited_sentences,
            "model": answer.model, "cached": answer.cached,
            "latency_ms": round(answer.latency_ms, 1),
            "sources": [{"n": c.n, "message_id": c.message_id, "sender": c.sender,
                         "date": c.date, "subject": c.subject, "text": c.text}
                        for c in answer.citations],
            "retrieval_ms": round(result.timings.get("retrieval_ms", 0), 1),
        })
        state = "refused" if answer.refused else f"cited {answer.cited_numbers}"
        print(f"  {i:3d}/{len(queries)}  [{q.query_class:12s}] {state:22s} "
              f"{q.query[:44]}", flush=True)
    return out


def rebuild_answer(row: dict):
    from emailrag.generation.synthesize import Answer, Citation

    return Answer(
        question=row["query"], text=row["answer"],
        citations=[Citation(n=s["n"], message_id=s["message_id"], sender=s["sender"],
                            date=s["date"], subject=s["subject"], text=s["text"])
                   for s in row["sources"]],
        cited_numbers=row.get("cited", []),
        invalid_citations=row.get("invalid_citations", []),
        uncited_sentences=row.get("uncited_sentences", []),
        refused=row.get("refused", False), model=row.get("model", ""),
    )


def judge_all(rows: list[dict], judge: GenerationJudge, skip_relevance: bool
              ) -> GenerationReport:
    report = GenerationReport(judge_model=judge.judge_model)
    for i, row in enumerate(rows, start=1):
        answer = rebuild_answer(row)
        score = judge.score(answer, question=row["query"],
                            judge_relevance=not skip_relevance)
        report.scores.append(score)

        unanswerable = row["query_class"] == "unanswerable"
        report.n_unanswerable += int(unanswerable)
        if row.get("refused"):
            if unanswerable:
                report.n_refused_on_unanswerable += 1
            else:
                report.n_refused_on_answerable += 1

        row["verdicts"] = [{"sentence": c.sentence, "cited": c.cited,
                            "verdict": c.verdict, "why": c.why}
                           for c in score.claims]
        row["groundedness"] = score.groundedness
        row["citation_accuracy"] = score.citation_accuracy
        row["relevance"] = score.relevance
        g = score.groundedness
        print(f"  {i:3d}/{len(rows)}  grounded="
              f"{'—' if g is None else f'{g:.2f}'}  "
              f"cites={score.citation_accuracy if score.citation_accuracy is None else round(score.citation_accuracy, 2)}  "
              f"rel={score.relevance}  {row['query'][:40]}", flush=True)
    return report


def compute_kappa(rows: list[dict], labels_path: Path) -> float | None:
    """Cohen's κ between the judge's verdicts and hand labels.

    Matched on (query_id, sentence): a label written against a different answer is
    not a label of this judge's verdict, and pairing them by position would silently
    compare unrelated items.
    """
    labels: dict[tuple[str, str], str] = {}
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        labels[(rec["query_id"], rec["sentence"].strip())] = rec["verdict"]

    judge_labels, human_labels, unmatched = [], [], 0
    for row in rows:
        for v in row.get("verdicts", []):
            key = (row["query_id"], v["sentence"].strip())
            if key in labels:
                judge_labels.append(v["verdict"])
                human_labels.append(labels[key])
            else:
                unmatched += 1

    print(f"\ncalibration: {len(judge_labels)} verdicts matched to hand labels "
          f"({unmatched} judge verdicts unlabelled)")
    if len(judge_labels) < 20:
        print("  too few matched labels for a meaningful kappa - the plan asks for "
              "50 hand-labelled answers", file=sys.stderr)
        return None
    return cohens_kappa(judge_labels, human_labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--index-root", type=Path, default=Path("data/index"))
    ap.add_argument("--sample", type=Path, default=Path("data/interim/sample.parquet"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--rerank", default=DEFAULT_RERANK)
    ap.add_argument("--transform", default="none")
    ap.add_argument("--n-sources", type=int, default=6)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--answers", type=Path, default=None,
                    help="score an existing answers file instead of generating")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--skip-relevance", action="store_true",
                    help="one fewer judge call per answer")
    ap.add_argument("--kappa", type=Path, default=None,
                    help="hand labels to calibrate the judge against")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.answers:
        rows = [json.loads(l) for l in args.answers.read_text().splitlines() if l.strip()]
        print(f"scoring {len(rows)} existing answers from {args.answers}")
        answers_path = args.answers
    else:
        if not args.eval.exists():
            print(f"error: {args.eval} missing - label some queries first:\n"
                  f"  make verify", file=sys.stderr)
            return 1
        queries = [q for q in load_eval(args.eval) if q.verified]
        if args.limit:
            queries = queries[:args.limit]
        if not queries:
            print("error: no verified queries in the eval set", file=sys.stderr)
            return 1

        index = args.index or default_index(args.index_root)
        if index is None:
            print(f"error: no built index under {args.index_root}", file=sys.stderr)
            return 1

        pipe = Pipeline(index, args.sample, rerank=args.rerank,
                        transform=args.transform)
        print(f"\ngenerating {len(queries)} answers ...")
        t0 = time.time()
        rows = generate(pipe, queries, args.n_sources)
        print(f"  {time.time()-t0:.0f}s")

        answers_path = args.out / f"answers_{stamp}.jsonl"
        answers_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"wrote {answers_path}")

    if args.generate_only:
        return 0

    judge = GenerationJudge(model=args.judge_model)
    print(f"\njudging with {judge.judge_model} "
          f"(a different client from the generator, deliberately) ...")
    report = judge_all(rows, judge, args.skip_relevance)

    kappa = None
    if args.kappa and args.kappa.exists():
        kappa = compute_kappa(rows, args.kappa)
    elif args.kappa:
        print(f"\nwarning: {args.kappa} does not exist - reporting uncalibrated",
              file=sys.stderr)

    print(f"\n{'='*70}")
    print(report.render(kappa=kappa))
    print(f"{'='*70}")

    scored_path = args.out / f"generation_{stamp}.json"
    scored_path.write_text(json.dumps({
        "generated_utc": stamp,
        "answers_file": str(answers_path),
        "judge_model": judge.judge_model,
        "kappa": kappa,
        "calibrated": kappa is not None,
        "metrics": {
            "groundedness": report.groundedness,
            "citation_accuracy": report.citation_accuracy,
            "relevance": report.relevance,
            "refusal_rate_on_controls": report.refusal_rate,
            "fabricated_citation_rate": report.fabricated_citation_rate,
            "n_unanswerable": report.n_unanswerable,
            "n_refused_on_answerable": report.n_refused_on_answerable,
        },
        "per_query": rows,
    }, indent=2))
    print(f"\nwrote {scored_path}")

    if kappa is None:
        print("\nNext: hand-label 50 answers' verdicts into a labels file and re-run "
              "with --kappa. Until then these numbers are one model's opinion of "
              "another's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
