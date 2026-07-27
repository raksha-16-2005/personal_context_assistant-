# What's left

Last updated: 27 July 2026

**Every module in the plan is now written. What remains is labels, compute, and
two human setup steps.** 458 tests, and they pass with torch and transformers
blocked so CI can run them without the pinned stack.

Nothing left on this list is blocked on code that does not exist:

| Blocked on | Items |
|---|---|
| `make verify` — ~1 h of human labelling | every quality table (dimensions 3–6, router accuracy, refusal rate) |
| `brew install ollama` | the extraction run, and the router's SQL arm having any data |
| a Kaggle account | 5 of 6 indices (~5 min on a T4 vs ~7.3 h locally) |
| hand-labelling 50 answers | Cohen's κ, without which judge scores are one model's opinion of another's |
| unattended machine time | the full 214k index, the overnight extraction pass |

---

## Blocking everything: label the pilot

```bash
make verify        # ~1 hour, 24 candidates already drafted
```

No ablation table, no router result, no generation eval exists until there are
labels. Nothing else on this list can produce a number without it.

Skip the three broken temporal candidates (`what is due mid next week`,
`what happened last Sunday`, `what are the WT Leg issues discussed this week`)
— they float against a "now" the corpus doesn't have. Fixed for the 80-query
run via `as_of`.

Then:

```bash
make bench         # first real tables
```

---

## Built and working

| | |
|---|---|
| Environment | venv, pinned Intel-macOS stack, pgvector 0.8.0 on pg15 |
| Corpus | 517,401 files → 214,282 messages → 50,065 sample |
| Chunking | all four strategies |
| Retrieval | dense (exact), BM25, RRF, weighted fusion |
| Metrics | recall@k, MRR, nDCG@10, per-class, latency |
| Eval set | schema, strict validator, candidate generator, pooled verify CLI |
| Harness | `run_ablation.py`, smoke-tested end to end |
| LLM client | Gemini/Groq/Anthropic/Ollama, `.env`, model rotation |
| LLM cache | content-addressed on disk, rotation-aware, on by default |
| Reranking | cross-encoder arms, head/tail contract, int8 ONNX export |
| Transforms | HyDE, multi-query, decomposition, with fired/degraded accounting |
| Failures | 6-category miss taxonomy + worklist |
| GPU offload | Kaggle notebook + parity-checked import path |
| Generation | cited answers, sentinel refusal, citation validation |
| Assembled system | `pipeline.py`, `make ask`, `make serve` (local web UI) |
| Index | 1 of 6 built (`thread_aware__all-MiniLM-L6-v2`, 57,997 chunks) |

## Not built yet

Ordered by the plan's weeks. "Compute" is unattended time on this machine.

### Week 1 — dimensions 1 & 2  ·  code done, needs compute

- `make index-d1` — 3 remaining chunkings × MiniLM · **~1.9 h**
- `make index-d2 BEST_CHUNKING=<D1 winner>` — bge-small + bge-base · **~5.4 h**
- Re-run `make bench`, paste tables into README

Nothing to write. Run overnight — **or ~5 min on a free Kaggle T4**, see
`notebooks/kaggle_build_indices.ipynb`. That path needs a phone-verified Kaggle
account and a private dataset holding `sample.parquet` plus `src/emailrag/`.

### Week 2 — dimensions 4, 5, 6  ·  code written, needs the eval set

All three are built and tested. The quality columns are empty only because
nothing is labelled yet; the latency columns are already measured.

- **Dimension 4, reranking.** `index/rerank.py`, `make bench-rerank`. Two
  findings came out of building it, both measured and both worth publishing:
  int8 ONNX is *slower* than torch fp32 on this CPU at realistic passage length
  (no AVX-512 VNNI), and **passage truncation** — not quantization — is what
  fits the 200 ms budget. `make bench-rerank-budget` has the grid; the only arm
  inside budget is L-2 @ top-20 with 192-token passages, at 158 ms.
- **Dimension 5, query transformation.** `query/transform.py`,
  `make bench-transform`. One LLM call per query per arm, all cached — a second
  run of the dimension-5 table makes zero network calls.
- **Dimension 6, failure analysis.** `evaluation/failures.py`, `make failures`.
  Six categories, not the plan's five: the residual had to split into
  `vocabulary` (few shared terms — expansion might help) and `ranking` (terms
  were there, ranked low anyway — expansion cannot help). `bad_label` is only
  ever set by hand.

### Week 3 — extraction and the router  ·  code written, needs Ollama

- **Prerequisite, and the only blocker:** `brew install ollama && ollama pull
  qwen2.5:3b`. CPU-only here at ~25 s/message, so `make extract` scopes to the
  threads the temporal and entity queries touch and the pre-filter drops most of
  even that. Until this runs, the router's SQL arm has no data — and it says so
  rather than reporting "nothing is due".
- **`extraction/`** — built. The design point: the model quotes the deadline
  phrase verbatim and **Python does the arithmetic**, because LLMs are unreliable
  at date maths and the error has to be recomputed to be caught. "next Thursday"
  is reported *ambiguous* with its alternative reading attached, since US usage
  does not converge and strict scoring would measure the annotator's convention.
- **`router/`** — built. Rules first (free, deterministic), model only on
  abstention, and abstention resolves to `both`: routing a date question to search
  gives a wrong answer, answering both ways is merely slower. `make route-eval`
  needs labels; the rules alone can be inspected with `--no-llm`.
- **`scripts/load_pgvector.py`** — built. `make pgvector` loads the pivot index
  and sweeps `ef_search` against the exact rankings, which turns HNSW recall loss
  from a hidden confound into a reported number. Needs Postgres running.

### Week 4 — generation  ·  synthesis built, measurement not

`generation/` exists and works: `make ask Q="..."` and `make serve` produce cited
answers over the pilot index. What is missing is the *eval*, and it splits into
two kinds of work.

Needs only code (`generation/judge.py`):

- Groundedness / faithfulness — does each claim follow from its cited source?
- Citation accuracy — does source [n] actually support the sentence citing it?
- Answer relevance — does the answer address the question asked?

Needs the eval set or a human:

- Refusal rate over the 10 unanswerable controls (needs the labelled set)
- **Judge calibration** — hand-label 50 answers, report Cohen's κ against the
  LLM judge. Without κ, a judge score is one model's opinion of another's.

Two structural checks already run on every answer and need no judge: citation
markers are validated against the sources supplied, and uncited assertions are
flagged. They catch only what is visible without reading the sources, which is
exactly why the judge work is still on this list.

### Week 5 — ship  ·  demo and CI built, tables need labels

- **HF Space demo** — built (`spaces/`), ZeroGPU path. Gradio there and stdlib
  `http.server` locally, because the Intel-macOS torch pin cannot tolerate
  Gradio's dependency floors. `check_public_corpus()` refuses to start unless
  Enron senders are present, so a private index cannot be served by accident.
- **CI** — built (`.github/workflows/ci.yml`). Tests plus eval-set validation,
  with no corpus and no keys; `index/embed.py`'s torch import is now lazy so the
  suite is importable without the pinned stack.
- **Privacy guard** — `make check-privacy`, meant as a pre-push hook.
- Still open: README and EVALUATION.md results sections, and one writeup. All
  three need numbers, which need labels.

---

## Known risks

**Gemini free-tier quota** was the tightest constraint on weeks 2–4 and is now
mitigated: `llm/cache.py` caches every call on disk, keyed on provider, model,
temperature, max_tokens and both prompts, and is on by default. Lookups are
rotation-aware, so a prompt answered by a fallback model yesterday is not re-paid
as a 429 today. Verified: a second `make bench-transform` over the same eval set
made **zero** network calls. `EMAILRAG_LLM_CACHE=0` forces a live run; `make
cache-clear` when you mean it.

The remaining quota exposure is the *first* run of each new arm, plus extraction
and the judge in weeks 3–4.

**Compute is the other one.** ~7.3 h of embedding for weeks 1's indices, plus
an overnight extraction run in week 3. Both unattended, but they serialise:
you get roughly one experiment cycle per day on this hardware.

**Thermal throttling is real and already measured** — sustained throughput is
2.6× below the microbenchmark. Plan against
[HARDWARE.md](HARDWARE.md)'s sustained column, not the burst column.
