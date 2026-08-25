# Email Retrieval System

A hybrid RAG pipeline over email, with a SQL/semantic query router.

**Live demo:** [huggingface.co/spaces/raksha-16-2005/email-retrieval-rag](https://huggingface.co/spaces/raksha-16-2005/email-retrieval-rag)
— public Enron corpus only, ZeroGPU. Latency there is not the latency reported
below; see the demo's own README for why.

**Live webapp:** [personal-context-assistant.onrender.com](https://personal-context-assistant.onrender.com)
— the multi-tenant Gmail RAG app (see [webapp/](webapp/)), sign in with your
own Google account. Free-tier Render + Neon; see [DEPLOY.md](DEPLOY.md) for
how it's deployed (and why the frontend isn't split onto its own domain -
a real Safari login bug, not a preference).

Every number below is measured on one machine and reproducible with
`make bench`. The eval set is published in [`data/eval/`](data/eval/) so the
tables can be checked independently.

> **Status: pilot results.** Tables below are real, but measured on a 26-query
> pilot (24 answerable + 2 unanswerable) — not yet the full 80-query eval set
> the plan targets. Read differences under ~0.02 nDCG as noise; the pilot's
> purpose is to catch harness bugs before scaling the eval set, not to be the
> final word. Empty cells are honest — nothing here is a placeholder for a
> number that was assumed.

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

Retriever fixed to hybrid-weighted (the dimension-3 winner) so chunking varies alone.

| Strategy | R@5 | R@20 | MRR | nDCG@10 |
|---|---|---|---|---|
| Fixed 512 | 0.605 | 0.812 | 0.471 | **0.499** |
| Fixed 512 + 64 overlap | 0.508 | 0.760 | 0.482 | 0.427 |
| Thread-aware | 0.591 | 0.715 | 0.427 | 0.478 |
| Whole message | 0.487 | 0.715 | 0.424 | 0.409 |

`fixed_512` edges out `thread_aware` here, but on 24 answerable queries that
0.02 gap is inside the noise floor — not a result yet, a direction to
re-check once the eval set is bigger.

### Dimension 2 — Embedding model

| Model | dim | R@20 | nDCG@10 | index MB | encode docs/s |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | **0.715** | **0.478** | 88.7 | 67.2 |
| bge-small-en-v1.5 | 384 | 0.471 | 0.326 | 88.7 | 34.5 |
| bge-base-en-v1.5 | 768 | 0.478 | 0.331 | 177.5 | 9.8 |

**Surprising, and held loosely: both bge models score well below MiniLM here,**
despite bge generally outperforming MiniLM on published retrieval
benchmarks. Query-side instructions are applied correctly for both bge
models (`embed.py`'s `QUERY_INSTRUCTION`), so this isn't the obvious
instruction-prefix bug. Candidate explanations not yet checked: bge's
training distribution may generalise worse to messy forwarded-email text
than MiniLM's, or 26 queries may simply be too few to trust a 0.15-point
nDCG gap. Reported as measured, not as a verified conclusion — re-check once
the eval set is bigger before citing this either way.

### Dimension 3 — Retriever

Pivot config: `thread_aware` / `all-MiniLM-L6-v2`.

| Configuration | R@5 | R@20 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|---|
| BM25 only | 0.417 | 0.467 | 0.441 | 0.391 | 4 |
| Dense only | 0.454 | 0.572 | 0.535 | 0.491 | 30 |
| Hybrid — RRF | 0.386 | 0.624 | 0.345 | 0.365 | 32 |
| Hybrid — weighted | **0.591** | **0.715** | 0.427 | **0.478** | 30 |

Weighted fusion beats every single-retriever baseline, on every chunking
strategy tried in dimension 1 — the one consistent story in this pilot.

### Dimension 4 — Reranking

Code and arms are built; the quality columns need a labelled eval set. The
latency column is already measured, and it decided the arms.

| Rerank arm | nDCG@10 | ΔnDCG | rerank p50 ms | in 200 ms budget |
|---|---|---|---|---|
| none (baseline) | 0.365 | — | — | yes |
| L-6 @ top-20, 512 tok | 0.348 | -0.018 | 628 | no |
| L-6 @ top-20, 192 tok | 0.347 | -0.018 | 381 | no |
| L-2 @ top-20, 192 tok | 0.326 | -0.039 | 158 | **yes** |
| L-6 @ top-50, 512 tok *(offline ceiling)* | | | 2,730 | no |

**Pilot finding, held loosely: every rerank arm *lowers* nDCG@10 here.**
On 24 answerable queries this is as likely to be noise as signal — the
ΔnDCG spread (-0.018 to -0.039) is inside what a bigger eval set could easily
flip. Worth re-running once the eval set scales past the pilot; not worth
concluding anything from yet. (Rerank p50 figures above are the
previously-measured clean baseline; this pilot's own bench-rerank run
happened while `index-d2` was building concurrently in the background, which
inflates raw latency — quality metrics are unaffected by that contention,
latency numbers from a loaded machine are not comparable to an idle one.)

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
| none (baseline) | — | — | 1 | 0.365 | — | 0 |
| HyDE | 17/26 | 9 | 1 | 0.338 | -0.027 | 17 |
| Multi-query expansion | 26/26 | 0 | 4 | 0.319 | -0.046 | 26 |
| Decomposition | 7/26 | 18 | 4 | 0.370 | +0.005 | 8 |

None of the three clearly help on this pilot — decompose is roughly flat
(and fired on only 7 of 26 queries), HyDE and multi-query both hurt. Same
noise caveat as dimension 4 applies; 26 queries is thin.

`fired` and `degraded` are columns, not footnotes: a transform that quietly fell
back to the original query on a third of the eval set would otherwise report the
baseline's numbers under its own name.

### Dimension 6 — Where this fails

Every recall@20 miss is classified into one of six categories, from the results
file, with no retrieval re-run: `bad_label`, `chunk_boundary`, `temporal`,
`multi_hop`, `vocabulary`, `ranking`. `make failures` prints the counts plus a
worklist with the evidence for each. Counts land here once the eval set exists.

**Pilot (26 queries): 6 of 24 answerable queries miss at recall@20** — 1
`temporal` (a "relative to" phrasing ranked at #26-27), 5 `vocabulary` (few
shared terms between query and message). No `chunk_boundary`, `multi_hop`, or
`ranking` misses showed up in this small a sample. `bad_label` is never
assigned automatically, per the rule above.

`bad_label` is the one category never assigned automatically — a system that can
file its own failures as mislabelled is grading its own exam.

### Generation

Answer synthesis with citations is **built and working** (`make serve`,
`make ask`), and so is the judge (`generation/judge.py`, `make
eval-generation`) — but a bug meant the first two attempts at this table
measured a quota outage instead of the system: the judge pinned an explicit
model name, which (by `llm/client.py`'s own design) disables the automatic
Gemini quota-rotation every other caller in this codebase gets, so one 429
failed every remaining verdict "unsupported" rather than falling through to
`gemini-3.5-flash`. Fixed in `judge.py`; numbers below are from the run after
the fix, on the 26-query pilot:

| Metric | Value | n |
|---|---|---|
| Groundedness | 0.088 | 36 claims |
| Citation accuracy | 0.088 | 36 cited claims |
| Answer relevance | 1.000 | 2 answers |
| Refusal rate on unanswerable controls | 1.000 | 2 controls |
| Fabricated-citation rate | 0.000 | 17 answers |
| Refused an answerable query | 7 | 24 answerable |

**Uncalibrated** — no hand labels have been compared against the judge yet,
so read groundedness/citation accuracy as one model's opinion of another's,
not a settled number. Two things worth flagging rather than quietly
reporting: groundedness at 0.088 is low even for a thin sample, and 7 of 24
answerable queries were refused this run against only 1 in an earlier run on
the same eval set and config — worth a closer look before this table is
treated as final. Answer relevance and refusal-on-controls both look
correct (2/2 controls now correctly refused, up from 0/2 previously) but
rest on an n of 2.

Cohen's κ against 50 hand-labelled verdicts is the next step and is
deliberately not faked here — see `generation/judge.py`'s own module
docstring: *"an uncalibrated instrument reports precision it has not
earned."* `--kappa` on `eval_generation.py` computes it once
`data/eval/answer_labels.jsonl` exists.

Two structural checks run on every answer without a judge: citation markers
are validated against the sources actually supplied, so a fabricated `[7]`
is surfaced instead of rendered as a working link; and sentences that assert
something without citing anything are flagged. Neither is a substitute for a
groundedness metric — they only catch the failures visible without reading
the sources.

### Router

Temporal/aggregate questions route to SQL over extracted commitments, semantic
questions route to hybrid search, and abstention resolves to `both` — see
[the plan](plan.html) for why. Pilot numbers, 24 scoreable queries (the 2
unanswerable controls excluded — there is no right arm for a question with no
answer):

| Query class | n | routed correctly |
|---|---|---|
| semantic | 8 | 1.000 |
| entity | 7 | 1.000 |
| temporal | 9 | 0.667 |
| **overall** | 24 | **0.875** |

Decided by rules 58% of the time, an LLM call on abstention the other 42%,
default-`both` 0%. The three temporal misses all had genuine date content but
got routed to `hybrid` instead of `sql`/`both` — content-vs-commitment framing
("who sent X on date Y" reads as a content lookup, not a commitment query) is
where the router's rules and its LLM fallback both currently agree, and agree
wrong. That the router's weak point is exactly the class it exists to catch is
the pilot's most useful finding so far, small as the sample is.

### Serving: pgvector / HNSW recall loss

Ablations above use exact search deliberately (see design decisions below);
the served system runs HNSW via pgvector, and this is the gap between them,
swept over `ef_search` on the pivot config:

| ef_search | recall@20 vs exact | worst query | top-1 agreement | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 40 | 0.792 | 0.100 | 0.885 | 1.8 | 2.7 |
| 100 | 0.917 | 0.700 | 0.923 | 2.2 | 2.6 |
| 200 | 0.962 | 0.750 | 0.923 | 3.6 | 4.5 |
| 400 | 0.989 | 0.950 | 0.962 | 5.7 | 6.8 |

At `ef_search=40` one query in this pilot loses 90% of its exact-search
recall while the average looks fine (0.792) — the mean hides it. `ef_search=200`
is the first setting where even the worst query holds onto most of its
recall, at a p95 of 4.5 ms; still well under the 200 ms generation budget.

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
