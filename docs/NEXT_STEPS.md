# What's left

Last updated: 27 July 2026

**Week 0 is ~85% done. Week 2's code is now written; weeks 1, 3–5 still need
compute or new modules.** The foundation and dimensions 4–6 are built and tested
(206 tests). What remains is the human labelling step, compute time, and the
extraction/router/generation modules.

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

### Week 3 — extraction and the router  ·  all new code

- **Prerequisite:** `brew install ollama && ollama pull qwen2.5:3b`. Not
  installed. CPU-only on this machine, ~25 s/message — scope to the ~2–3k
  messages the temporal and entity queries touch, run overnight.
- **`extraction/`** — commitments schema, prompts, relative-date resolution
  (the metric that matters: "next Thursday" sent on a Tuesday → absolute UTC).
  Two models: Qwen 2.5 3B local vs Claude Haiku 4.5 as the ceiling (~$3).
- **`router/`** — classify temporal/aggregate → SQL, semantic → hybrid,
  ambiguous → both. Report router accuracy *and* end-to-end quality per class.
- **`scripts/load_pgvector.py`** — `index/store.py` exists but nothing calls
  it. Needed for the served system and to measure HNSW recall loss against the
  exact baseline.

### Week 4 — generation eval  ·  all new code

- **`generation/`** — answer synthesis with citations. Does not exist at all;
  everything so far stops at retrieval.
- Groundedness, citation accuracy, refusal rate on the 10 unanswerable controls,
  answer relevance.
- **Judge calibration** — hand-label 50 answers, report Cohen's κ against the
  LLM judge.

### Week 5 — ship

- HF Space demo. **Must use the ZeroGPU path** — plain CPU Gradio Spaces are no
  longer free; free personal accounts get 2 ZeroGPU Spaces.
- README with all tables, EVALUATION.md results sections filled in.
- One writeup on the router or chunking finding.

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
