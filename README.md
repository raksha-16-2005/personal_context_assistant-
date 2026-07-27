# Email Retrieval System

A hybrid RAG pipeline over email, with a SQL/semantic query router.

Every number below is measured on one machine and reproducible with
`make bench`. The eval set is published in [`data/eval/`](data/eval/) so the
tables can be checked independently.

> **Status: in progress.** Tables are empty until the corresponding stage
> lands. Empty cells are honest — nothing here is a placeholder for a number
> that was assumed.

## Try it

```bash
make serve     # http://127.0.0.1:8000
make ask Q="what was decided about the confidentiality agreement"
```

Ask a question, get an answer built only from retrieved emails, with every claim
carrying a clickable citation back to the message it came from. Retrieval is
hybrid (dense + BM25, reciprocal-rank fusion) with a cross-encoder rerank.

Measured on the pilot index (58k chunks over 50k messages), warm:

| stage | ms |
|---|---|
| retrieval (dense + BM25 + RRF) | ~20 |
| rerank (L-2 cross-encoder, top-20, 192 tok) | ~300 |
| generation (Gemini 2.5 Flash, 6 sources) | ~2,200 |
| **cached re-ask** | **~330** |

The two behaviours worth checking yourself:

**It refuses.** Asked *"who won the 2030 world cup"* it returns
`INSUFFICIENT_CONTEXT` — even though retrieval does surface soccer newsletters as
distractors. Asked *"what is the capital of France"* it answered *"Paris is a city
in France [5]. The excerpts do not explicitly state that Paris is the capital of
France."* — declining to fill the gap from its own weights, which is the whole
point.

**It distinguishes.** Asked about "the confidentiality agreement" over a corpus
containing dozens, it separates four different ones and cites each.

The UI is `http.server` from the standard library and a single self-contained HTML
page — no new dependencies. That is deliberate: `requirements.txt` is pinned to
the last torch/transformers combination that works on Intel macOS, and both Gradio
and Streamlit require newer `huggingface_hub` and `pydantic` than that stack
tolerates. The public ZeroGPU Space in phase 6 is where Gradio belongs, in a fresh
environment where the pin does not apply.

## Results

### Dimension 1 — Chunking

| Strategy | R@5 | R@20 | MRR | nDCG@10 |
|---|---|---|---|---|
| Fixed 512 | | | | |
| Fixed 512 + 64 overlap | | | | |
| Thread-aware | | | | |
| Whole message | | | | |

### Dimension 2 — Embedding model

| Model | dim | R@20 | nDCG@10 | index MB | encode docs/s |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | | | | 67.2 |
| bge-small-en-v1.5 | 384 | | | | 34.5 |
| bge-base-en-v1.5 | 768 | | | | 9.8 |

### Dimension 3 — Retriever

| Configuration | R@5 | R@20 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|---|
| BM25 only | | | | | |
| Dense only | | | | | |
| Hybrid — RRF | | | | | |
| Hybrid — weighted | | | | | |

### Dimension 4 — Reranking

Code and arms are built; the quality columns need a labelled eval set. The
latency column is already measured, and it decided the arms.

| Rerank arm | nDCG@10 | ΔnDCG | rerank p50 ms | in 200 ms budget |
|---|---|---|---|---|
| none (baseline) | | — | — | yes |
| L-6 @ top-20, 512 tok | | | 628 | no |
| L-6 @ top-20, 192 tok | | | 381 | no |
| L-2 @ top-20, 192 tok | | | 158 | **yes** |
| L-6 @ top-50, 512 tok *(offline ceiling)* | | | 2,730 | no |

Two negative results came out of building this, both measured:

**int8 ONNX quantization does not buy the shipped path anything on this CPU.**
The export is real — 91 MB → 23 MB, rank correlation ρ=0.984 against fp32 — and
at short passage lengths it looks like a 1.6× win. At the corpus's actual mean
chunk length of ~300 tokens it is **slower than torch fp32** (760 ms vs 644 ms at
top-20). This machine has AVX2 but no VNNI, so the dynamic quantize/dequantize
overhead is never repaid by the integer kernels. Reproduce with `make
export-onnx`.

**Passage truncation, not quantization, is the lever that fits the budget.**
Attention is superlinear in sequence length, so scoring 192 tokens instead of 512
is 2–3× cheaper — and unlike lowering *k* it costs no candidates, since every
chunk still gets scored. `make bench-rerank-budget` sweeps model depth × k ×
passage length; the full grid is in `runs/rerank_budget.json`.

### Dimension 5 — Query transformation

| Transform | fired | degraded | runs/query | nDCG@10 | ΔnDCG | llm calls |
|---|---|---|---|---|---|---|
| none (baseline) | — | — | 1 | | — | 0 |
| HyDE | | | 1 | | | 1/query |
| Multi-query expansion | | | 4 | | | 1/query |
| Decomposition | | | 1–4 | | | 1/query |

`fired` and `degraded` are columns, not footnotes: a transform that quietly fell
back to the original query on a third of the eval set would otherwise report the
baseline's numbers under its own name.

### Dimension 6 — Where this fails

Every recall@20 miss is classified into one of six categories, from the results
file, with no retrieval re-run: `bad_label`, `chunk_boundary`, `temporal`,
`multi_hop`, `vocabulary`, `ranking`. `make failures` prints the counts plus a
worklist with the evidence for each. Counts land here once the eval set exists.

`bad_label` is the one category never assigned automatically — a system that can
file its own failures as mislabelled is grading its own exam.

### Generation

Answer synthesis with citations is **built and working** (`make serve`,
`make ask`). What is not yet built is the *measurement* of it:

| Metric | status |
|---|---|
| Refusal rate on the 10 unanswerable controls | needs the labelled eval set |
| Groundedness / faithfulness | needs a judge (`generation/judge.py`) |
| Citation accuracy | needs a judge |
| Answer relevance | needs a judge |
| Cohen's κ, judge vs 50 hand-labelled answers | needs the hand labels |

Two structural checks run on every answer today, without a judge: citation
markers are validated against the sources actually supplied, so a fabricated
`[7]` is surfaced instead of rendered as a working link; and sentences that
assert something without citing anything are flagged. Neither is a substitute for
a groundedness metric — they only catch the failures visible without reading the
sources.

### Router

Not yet built. See [the plan](plan.html).

## Design decisions worth arguing with

**Metrics are computed at message level, not chunk level.** The eval set
labels which *messages* answer a query. Scoring chunks would let a strategy
that emits four chunks per message outscore one emitting a single chunk purely
by occupying more of the top-k — an artifact of granularity, not retrieval
quality, and it would make dimension 1 uninterpretable.

**Ablations use exact search, not HNSW.** Approximate search has its own
recall loss that varies with dimensionality and how clustered a model's
embedding space is. An HNSW-backed comparison of bge-base against MiniLM would
measure the model *and* the index while reporting it as a property of the
model. pgvector/HNSW backs the served system, where the approximation loss is
measured against these exact numbers rather than hidden inside them.

**Unanswerable controls are excluded from recall/MRR/nDCG, not scored as
zero.** There is no correct document to retrieve, so the metrics are undefined.
Scoring them zero would drag every mean down by 12.5% and make these tables
incomparable to any published baseline. They are measured separately, on
refusal rate, in the generation eval.

**One reference tokenizer decides all chunk boundaries.** Chunking with each
model's own tokenizer would make dimensions 1 and 2 interact when they are
supposed to vary independently.

**Reranking reorders the head and keeps the tail.** Truncating the ranking at *k*
would shrink the number of distinct messages available to recall@20, so a
reranker that reordered the head perfectly could still post a recall drop caused
entirely by truncation. Dimension 4 measures reordering, and only reordering.

**Multi-text transforms fuse within a modality before across.** Four query
rewrites must not outvote the sparse retriever 4:1 inside a "50/50" hybrid —
that would make a dimension-5 change silently a dimension-3 change too. So the
rewrites are fused by RRF first, then the dense and sparse sides are combined by
whatever dimension 3 specifies.

**HyDE rewrites the dense side only.** An invented `From: sara.chen@company.com`
is useful to a bi-encoder and pure noise to BM25, which would match the
fabricated name literally. Applying one rewrite to both retrievers would conflate
"does HyDE help" with "does HyDE help BM25", and those have different answers.

**Chunk text is rebuilt from the corpus, not stored twice.** Reranking needs
passages, and persisting them beside six indices would add ~700 MB of second
copies. The rebuild is exact — chunk ids are `{dedup_key}:{ordinal}` and
selecting *whole threads* reproduces byte-identical chunks — and it is verified
against the index's own chunk ids on every load rather than trusted.

**Threading is two-pass, and the second pass is a heuristic.** Enron's
JavaMail export left a large share of messages with a `Message-ID` but no
reply headers, so a references-only threader leaves most of the corpus as
singletons and the thread-aware arm tests nothing. Pass 2 unions on normalized
subject **and** participant overlap **and** a 90-day window; subject alone
merges years of unrelated "Re: meeting". `sample.stats.json` reports how much
threading came from each pass.

## Hardware

Built and measured on a 2019 MacBook Pro (i7-9750H, 16 GB, **no usable GPU**).
That constraint is not incidental — it changed the plan in four places
(subsample size, reranker choice, extraction scope, where Postgres runs).
[`docs/HARDWARE.md`](docs/HARDWARE.md) documents what was measured and what
each measurement forced.

Two constraints that will bite anyone reproducing this on an Intel Mac:

- PyTorch ships no `macosx_x86_64` wheel after **2.2.2**, and
  `transformers >= 4.46` silently disables the torch backend against it.
  `requirements.txt` pins the newest working combination.
- Use Homebrew's `python3.11`, not pyenv's — the pyenv build here lacks
  `liblzma`, which breaks `datasets` and therefore `sentence-transformers`.

## Corpus

```
517,401 files  ->  255,498 unique  ->  214,282 indexable  ->  50,065 sample
```

Three things the plan assumed about this corpus were wrong, and all three would
have quietly corrupted results. [`docs/CORPUS.md`](docs/CORPUS.md) has the
detail; the short version:

- **`Message-ID` deduplicates nothing here.** The JavaMail export stamped a
  fresh id on every copy, so all 435k rows of a first pass had distinct ids.
  Content hashing collapses **50.6%** of the corpus.
- **`In-Reply-To` and `References` are entirely absent** — not sparse, absent.
  All threading is reconstructed heuristically, which is a caveat on every
  thread-aware result.
- **"~80% of mail is bulk" is a Gmail-era figure.** Measured here: 14.2%.

## Quickstart

```bash
make venv          # pinned Intel-macOS stack
make download      # CMU Enron corpus, 443 MB
make extract       # ~2.6 GB maildir
make corpus        # parse -> content dedup -> bulk + blast filters
make sample        # thread reconstruction + stratified subsample
make index-pilot   # one index, enough to start labelling (~15 min)
make candidates    # draft eval queries        (needs GEMINI_API_KEY)
make verify        # hand-verify them          (this is the human step)
make bench         # reproduce every table
```

Then the per-dimension tables:

```bash
make bench-rerank         # dimension 4: arms vs added latency
make bench-transform      # dimension 5: HyDE / multi-query / decomposition
make failures             # dimension 6: classify every recall@20 miss
make bench-rerank-budget  # which rerank arm fits 200 ms on your CPU
make export-onnx          # int8 export + a rank-correlation check
```

`make index-all` builds the full six-config matrix — an overnight job on this
hardware, or **~5 minutes on a free Kaggle T4** via
[`notebooks/kaggle_build_indices.ipynb`](notebooks/kaggle_build_indices.ipynb)
plus `scripts/import_embeddings.py`. `make bench-hardware` re-measures throughput
into `runs/hardware.json`.

**Every LLM call is cached to disk** (`data/llm_cache/`, gitignored). Dimension 5,
the router and the judge each call an LLM per query per config; without a cache a
single `make bench` re-spends the whole free-tier daily quota and the tables
cannot be regenerated the same day. A cached re-run of dimension 5 makes **zero**
network calls. `make cache-stats` / `make cache-clear`; `EMAILRAG_LLM_CACHE=0`
forces a live run.

### Offloading embedding to a GPU, honestly

> Quality metrics (recall, MRR, nDCG) are **hardware-independent** — same
> weights, same math. Compute them wherever is fastest.
> Latency metrics are **hardware-specific**. Measure them on one stated platform
> and never mix platforms within a table.

`import_embeddings.py` re-chunks the local corpus and refuses to import unless
every chunk id matches the notebook's, **in order** — row *i* of the matrix is
chunk *i* and nothing else records that pairing. It also rejects a matrix whose
rows are not unit length, since `DenseIndex` treats the dot product *as* the
cosine. Each imported `config.json` records `embed_platform`, so a T4 throughput
figure can never end up in a table of CPU numbers.

## Layout

```
src/emailrag/
  corpus/     maildir parsing, dedup, bulk + blast filters, threading, sampling
  chunking/   the four chunking strategies
  index/      dense (exact), sparse (BM25), fusion, reranking, pgvector store
  query/      dimension-5 transforms (HyDE, multi-query, decomposition)
  generation/ answer synthesis with validated citations
  evaluation/ IR metrics, eval-set schema, ablation harness, failure taxonomy
  llm/        provider-agnostic client + on-disk response cache
  ui/         the single-page local UI
  pipeline.py the assembled system: question -> cited answer
scripts/      pipeline entry points (ask.py, serve.py, run_ablation.py, …)
notebooks/    Kaggle GPU embedding notebook
docs/         CORPUS.md, EVALUATION.md, HARDWARE.md, NEXT_STEPS.md
data/eval/    the published eval set
```

`evaluation/harness.py` and `pipeline.py` are deliberately separate. The harness
runs 80 queries through many configurations and reports metrics; it holds no
message metadata because metrics never need it. The pipeline runs one question
through one configuration and must produce something a person can read — sender,
date, subject, excerpt. They share every component, so the configuration that
wins the ablation is the configuration the UI serves.
