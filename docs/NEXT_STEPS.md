# What's left

Last updated: 5 August 2026

**Every module in the plan is written, every dimension-1/2 index is built,
and the pilot has real numbers end to end** — retrieval, reranking,
transforms, router, generation, and pgvector's HNSW recall loss. 458 tests
pass with torch and transformers blocked so CI can run without the pinned
stack.

What's left is one deliberate scope decision, not a blocker:

| Blocked on | Items |
|---|---|
| ~2.5–3 h of careful (not rushed) hand-labelling | scaling the eval set from 26 to the plan's 80 queries |
| ~1–1.7 h of careful hand-labelling | Cohen's κ — 50 hand-labelled answer verdicts against the judge |
| a Kaggle account, if repeating this elsewhere | 5 of 6 dimension-1/2 indices (already built locally here — ~5 min on a T4 vs ~7.3 h locally) |
| unattended machine time, if wanted | the full 214k-message index, an overnight full-corpus extraction pass |

Both hand-labelling items are **deliberately deferred, not stalled.**
`data/eval/queries.jsonl.stale-2026-08-02-rushed` is a live example from this
same project of what a rushed pass produces: five straight entity queries
marked `verified: true` with zero relevant messages, including one whose
pool clearly contains a real answer. It was caught and discarded. Labels
produced under time pressure are worse than no labels — they cost the
re-verification time later *and* poison anything computed from them in the
meantime. Both are scoped and ready to resume (100 candidates drafted,
44/30/14/12 across semantic/temporal/unanswerable/entity — comfortably over
target; the generation answers file for κ already exists), just not rushed
tonight.

---

## Built and working

| | |
|---|---|
| Environment | venv, pinned Intel-macOS stack, pgvector 0.8.0 on pg15 |
| Corpus | 517,401 files → 214,282 messages → 50,065 sample |
| Chunking | all four strategies, all built |
| Retrieval | dense (exact), BM25, RRF, weighted fusion |
| Metrics | recall@k, MRR, nDCG@10, per-class, latency |
| Eval set | 26 verified queries (pilot), 100 candidates drafted for the next pass |
| Harness | `run_ablation.py`, `load_pgvector.py` — both run end to end on real data |
| LLM client | Gemini/Groq/Anthropic/Ollama, `.env`, model rotation |
| LLM cache | content-addressed on disk, rotation-aware, on by default |
| Reranking | cross-encoder arms, head/tail contract, int8 ONNX export |
| Transforms | HyDE, multi-query, decomposition, with fired/degraded accounting |
| Failures | 6-category miss taxonomy + worklist |
| Extraction | scoped Ollama run over eval threads: 13 commitments, router's SQL arm has data |
| Router | pilot accuracy 0.875 (24 scoreable queries); temporal is the weak point — see README |
| Generation | cited answers, sentinel refusal, citation validation, **and a working judge** |
| Judge | fixed tonight — see "Fixed tonight" below |
| pgvector | pivot index loaded, HNSW recall swept against exact search — see README |
| Indices | **all 6 built**: 4 chunkings × MiniLM, + bge-small and bge-base on `thread_aware` |

## Fixed tonight

**`generation/judge.py` pinned an explicit model name** (`gemini-2.5-flash`),
which disables `llm/client.py`'s automatic quota-rotation — that rotation
only activates when a caller passes `model=None`. Every other caller in this
codebase (`synthesize.py`'s generator, `make_eval_candidates.py`) already
does this correctly; the judge didn't. The result: the first `make
eval-generation` run reported groundedness = 0.000 and citation accuracy =
0.000 across the board, which read as a broken system but was actually a
single exhausted quota killing every remaining verdict instead of falling
through to `gemini-3.5-flash` or `gemini-flash-lite-latest`. Fixed by
leaving the judge's model unpinned unless `--judge-model` is passed
explicitly. Real numbers are in now — see README's Generation section. They
are low (0.088 groundedness) and one behavior changed between runs (7
refusals on answerable queries vs. 1 previously, on the same config) — both
flagged in the README rather than smoothed over, since the pilot is too
thin to tell noise from a real regression yet.

## Not built yet

### Scale the eval set: 26 → 80 queries

100 candidates are drafted (target stratification is 35 semantic / 25
temporal / 10 entity / 10 unanswerable; currently at 8/9/7/2 verified).
`make verify` resumes from `data/eval/queries.jsonl` and only presents what
isn't yet verified. Budget ~2.5–3 h based on this project's own measured
rate (~2.5 min/candidate). Do not rush it — see above.

### Cohen's κ

`runs/generation_20260805T152807Z.json` has the judge's per-claim verdicts
already. Hand-label the same claims independently into
`data/eval/answer_labels.jsonl` (format: `{"query_id":, "sentence":,
"verdict":}` per line, per `eval_generation.py`'s own docstring), then:

```bash
python scripts/eval_generation.py --answers <answers file> --kappa data/eval/answer_labels.jsonl
```

### Re-run at 80-query scale, once labelled

`make bench`, `make route-eval`, `make eval-generation` — all mechanical
once the eval set is bigger, no new code needed. Budget ~30–60 min,
some of it unattended.

### Everything else in the original plan

- **Full 214k-message index** and **overnight full-corpus extraction** —
  production-scale, not needed for any table published so far. The
  eval-scoped extraction already covers what the router pilot needs.
- **HF Space demo, CI, privacy guard** — built, unchanged tonight.
- README/EVALUATION.md results sections — updated tonight through pilot
  scale; will need a second pass once the 80-query numbers exist.

---

## Known risks

**Gemini free-tier quota** is real but self-healing: it rotates
automatically now (judge included) across three models, and resets daily.
Tonight's candidate-drafting (31 calls) plus generation+judging (~150 calls)
ran into a burst limit once and recovered within minutes on retry — not the
multi-hour outage the earlier framing implied.

**Compute is no longer the constraint it was.** All six dimension-1/2
indices are built (bge-base finished at 03:17 this morning). The ~7.3 h of
unattended embedding this doc used to flag as the tightest scheduling
constraint is done. What's left is bounded by human labelling time, not
machine time.

**Thermal throttling is real and already measured** — sustained throughput
is 2.6× below the microbenchmark. Plan against
[HARDWARE.md](HARDWARE.md)'s sustained column, not the burst column.
