# TODO

`[you]` needs a human · `[me]` I can do it · `[compute]` unattended machine time

Times are measured or projected on this machine — see
[docs/HARDWARE.md](docs/HARDWARE.md).

---

## Phase 0 — finish the eval set (blocks literally everything)

**This is still the only thing standing between the repo and its first real
table.** Dimensions 4, 5 and 6 are now built and tested; their quality columns
are empty solely because nothing is labelled. `make verify` is the highest-value
hour available.

- [ ] **`[you]`** `make verify` — label the 24 drafted candidates · **~1 h**
      Skip the 3 broken temporal ones (they float against a "now" the corpus
      lacks; already fixed for the next batch via `as_of`).
- [ ] **`[me]`** `make bench` — first real tables from those labels
- [ ] **`[me]`** Review the pilot: is `--pool-k 30` deep enough? do the metrics
      look sane? is anything in the harness wrong before we scale?
- [ ] **`[me]`** `make candidates --n-per-class 10` for the full 80
      *(needs Gemini quota — likely a fresh day)*
- [ ] **`[you]`** Label the remaining ~56 · **~3 h**
- [ ] **`[me]`** `make validate-eval --strict`, publish `data/eval/queries.jsonl`

## Phase 1 — dimensions 1 & 2 (code is written, just needs running)

Two routes. **Kaggle is strongly preferred** — see Phase 1b. Local works but
costs ~7.3 h of throttled CPU instead of ~5 min of T4.

- [ ] **`[compute]`** `make index-d1` — 3 remaining chunkings × MiniLM · **~1.9 h**
- [ ] **`[me]`** `make bench --dimension chunking` → chunking table, pick winner
- [ ] **`[compute]`** `make index-d2 BEST_CHUNKING=<winner>` — bge-small +
      bge-base · **~5.4 h, overnight**
- [ ] **`[me]`** `make bench --dimension model` → size/quality/latency curve

**Phase 1 note.** `bge-reranker-base` is now a defined dimension-4 arm
(`bge@50` in `index/rerank.py`) rather than an unusable ceiling, but it is
opt-in: at 6.8 s/query it costs hours over a full eval set. Run it once, on the
winning config, when there is something to be the ceiling *of*.

## Phase 1b — offload embedding to Kaggle (free T4/P100)

Free tier: **~30 GPU h/week, 12 h sessions, T4 or P100 (16 GB)**. The models
here are 22M–109M-parameter encoders, so the GPU is idle-fast on them:
projected ~5 min for all six indices against ~7.3 h locally. It also promotes
`bge-reranker-base` from an unusable 6.8 s/query "offline ceiling" to a real
ablatable config, and takes extraction from ~25 s/message to ~1–2 s.

**The rule that keeps this honest — build to it from the start:**

> **Quality metrics (recall, MRR, nDCG) are hardware-independent** — same
> weights, same math — so compute them wherever is fastest.
> **Latency metrics are hardware-specific** — measure them on one stated
> platform and never mix platforms within a table.

Reported explicitly ("quality on T4, latency on a documented CPU baseline")
this is more rigorous than hiding it. Reported carelessly it is dishonest.

- [ ] **`[you]`** Kaggle account, phone-verify to enable notebook internet
      access (needed to pull HF model weights)
- [ ] **`[you]`** Upload a **private** Kaggle Dataset holding
      `data/interim/sample.parquet` (53 MB) **and** `src/emailrag/`. The package
      has to travel with the data — the notebook imports the repo's own
      `chunk_corpus` rather than a retyped copy, which is what makes chunk-id
      parity achievable at all. Enron is public data, so this is fine —
      **never upload Gmail-derived data.**
- [x] **`[me]`** `notebooks/kaggle_build_indices.ipynb` — modern CUDA torch (the
      2.2.2 pin is a macOS-x86 constraint only), chunk + embed all six configs,
      write `.npy` shards to `/kaggle/working`, self-check before saving
- [ ] **`[you]`** `Save Version`, then pull with
      `kaggle kernels output <user>/<slug> -p ~/Downloads/kaggle-emailrag`
- [x] **`[me]`** `scripts/import_embeddings.py` — drop downloaded shards into
      `data/index/<config>/` so `make bench` runs unchanged against them.
      Also builds the BM25 side locally (CPU-cheap, no GPU session needed).
- [x] **`[me]`** Assert chunk-id parity between local chunking and the notebook's,
      so a GPU-built index is provably the same corpus as a local one.
      `chunktext.verify_parity`, **ordered** — row *i* of the matrix is chunk *i*
      and nothing else records that pairing. Also rejects non-unit-length rows,
      since `DenseIndex` treats the dot product *as* the cosine.
- [x] **`[me]`** Measure latency **locally only**, and label every latency table
      with the CPU it came from. Imported configs carry `embed_platform`; the
      dimension 4/5 tables print the CPU caveat.
- [x] **`[me]`** Add `bge-reranker-base` as a real dimension-4 config now that
      it is affordable, keeping the CPU-latency caveat attached (`bge@50`,
      opt-in — 6.8 s/query)

## Phase 2 — dimensions 3, 4, 5

**Code complete. Every remaining box needs labels, not code.**

- [ ] **`[me]`** Dimension 3 (retriever) table — falls out of `make bench` free
- [x] **`[me]`** **LLM response cache. Do this FIRST.** `llm/cache.py`, on by
      default, rotation-aware, keyed on provider/model/temperature/max_tokens/
      both prompts. Verified: a second `make bench-transform` over the same eval
      set made **zero** network calls.
- [x] **`[me]`** `index/rerank.py` — cross-encoder over top-k. Reranks the head
      and keeps the tail: truncating at *k* would shrink the distinct-message
      count and post a recall drop caused entirely by truncation.
- [x] **`[me]`** ONNX int8 export — `scripts/export_onnx_reranker.py`, with a
      rank-correlation check against fp32 (ρ=0.984, top-5 overlap 5/5).
      **It does not get the shipped path under 200 ms.** At the corpus's real
      ~300-token chunk length int8 is *slower* than torch fp32 (760 vs 644 ms at
      top-20) — this CPU has AVX2 but no VNNI, so the dynamic quantize overhead
      is never repaid. Negative result, worth publishing.
- [x] **`[me]`** Find something that *does* fit 200 ms —
      `scripts/bench_rerank_budget.py` sweeps model depth × k × passage length.
      Truncation is the lever (512→128 tokens is 2–5×, and unlike lowering *k* it
      costs no candidates). Only arm inside budget: **L-2 @ top-20, 192 tokens,
      158 ms.**
- [ ] **`[me]`** Dimension 4 table — nDCG gain *against* added latency.
      `make bench-rerank`; the ΔnDCG and rerank-ms columns are already wired.
- [x] **`[me]`** Query transformation: HyDE, multi-query expansion, decomposition
      (`query/transform.py`). One call per query per arm, `fired`/`degraded`
      counted so a silent fallback can't publish the baseline under a
      transform's name.
- [ ] **`[me]`** Dimension 5 table — **publish the ones that lose**.
      `make bench-transform`.

## Phase 3 — dimension 6, failure analysis

- [x] **`[me]`** Categorise every recall@20 miss — `evaluation/failures.py`,
      `make failures`. **Six** categories, not five: the residual had to split
      into `vocabulary` (few shared terms, expansion might help) and `ranking`
      (terms were present, ranked low anyway — expansion cannot help, reordering
      might). Filing the second under "vocabulary mismatch" would point the fix
      at the wrong lever. `bad_label` is never assigned automatically — a system
      that can file its own failures as mislabelled is grading its own exam.
- [ ] **`[me]`** Fix the bad labels it finds, re-run bench.
      `make failures` prints a worklist; write `bad_label: <why>` into the
      query's `notes` and re-run.
- [ ] **`[me]`** "Where this fails" README section with the taxonomy and counts
      *(section scaffolded, counts need labels)*

## Phase 4 — extraction and the router

- [ ] **`[you]`** `brew install ollama && ollama pull qwen2.5:3b`
      *(CPU-only here, ~25 s/message)*
- [ ] **`[me]`** `commitments` table schema + migration
- [ ] **`[me]`** Extraction prompts + **relative-date resolution** — "next
      Thursday" sent on a Tuesday → absolute UTC. Top source of silent
      wrongness; gets its own accuracy metric.
- [ ] **`[compute]`** Run Qwen 2.5 3B over the ~2–3k messages the temporal and
      entity queries touch · **overnight**
- [ ] **`[me]`** Run Claude Haiku 4.5 as the quality ceiling · **~$3, batched**
- [ ] **`[me]`** Date-accuracy metric, local vs ceiling
- [ ] **`[me]`** `router/` — temporal/aggregate → SQL, semantic → hybrid,
      ambiguous → both and merge. `pipeline.py` is where it plugs in: `search()`
      currently always takes the hybrid path, and the router decides that instead.
- [ ] **`[me]`** Router accuracy table + end-to-end quality **per query class**
      (this is the headline result)
- [ ] **`[me]`** `scripts/load_pgvector.py` — `index/store.py` exists but
      nothing calls it yet
- [ ] **`[me]`** Measure HNSW recall loss against the exact baseline

## Phase 5 — generation eval

- [x] **`[me]`** `generation/` — answer synthesis with citations. Built and
      working end to end: `make ask Q="..."` and `make serve`. Refusal is a
      machine-checkable sentinel (`INSUFFICIENT_CONTEXT`) rather than
      pattern-matched prose, citations are validated against the sources actually
      supplied, and sources are messages rather than chunks so a citation points
      at something a person can open.
- [x] **`[me]`** `pipeline.py` + `ui/` — the assembled system and a local web UI
      over it. `http.server` and one self-contained HTML page, no new
      dependencies: Gradio and Streamlit both need newer `huggingface_hub` and
      `pydantic` than the Intel-macOS torch pin tolerates.
- [ ] **`[me]`** Groundedness / faithfulness metric
- [ ] **`[me]`** Citation accuracy metric
- [ ] **`[me]`** Refusal rate on the 10 unanswerable controls
- [ ] **`[me]`** Answer relevance metric
- [ ] **`[you]`** Hand-label 50 answers for judge calibration · **~1 h**
- [ ] **`[me]`** Cohen's κ, judge vs human

## Phase 6 — ship

- [x] **`[me]`** `git init`, first commit. Verified before committing that
      `.env`, `data/raw`, `data/interim`, `data/index`, `data/llm_cache`,
      `data/onnx` and `.venv` are all excluded — 64 files, source and docs only.
- [ ] **`[compute]`** Build the full 214k-message index for the winning config
      so the "500k emails" claim is honest · **overnight**
- [ ] **`[me]`** GitHub Actions CI — tests + `validate-eval` on push
- [ ] **`[me]`** HF Space demo. **ZeroGPU path only** — plain CPU Gradio Spaces
      are no longer free; free personal accounts get 2 ZeroGPU Spaces.
- [ ] **`[me]`** README with every table filled in, numbers before prose
- [ ] **`[me]`** EVALUATION.md results sections
- [ ] **`[you/me]`** One writeup on the router or chunking finding

## Phase 7 — point it at your own Gmail

Off the critical path, and deliberately last. The Enron half is what makes this
**shareable**: a demo over your private mail is one nobody else can run,
reproduce, or verify. Build the benchmark first, then aim the winning config at
your own mailbox — the pipeline is corpus-agnostic by design, and Gmail's API
returns RFC822 messages, which is exactly what `corpus/enron.py` already parses.

### Design decisions to build to, not discover later

**Retrieval over all history; extraction over recent mail only.** Indexing is
cheap and searching old mail is the point. Extraction is not: Ollama is CPU-only
here at ~25 s/message, so a 35k-message mailbox would be ~240 hours. Cap it at
roughly the last 90 days and run incrementally after that. You do not care what
was due in 2019.

**Pre-filter before the extractor.** Most messages contain no commitment at all.
A cheap date/deadline regex skips them before the LLM sees them — same reasoning
as the bulk filter, and it is the difference between an overnight run and a
week.

**Keep extraction local.** Beyond cost, `qwen2.5:3b` via Ollama keeps your
actual message bodies on your machine. Retrieval embeddings are already local;
sending private mail to Gemini or Anthropic for extraction is a real choice, not
a default.

### Tasks

- [ ] **`[you]`** Google Cloud project + OAuth client, consent screen in
      **Testing**, scope `gmail.readonly`
- [ ] **`[you]`** Understand the catch: testing-status apps get refresh tokens
      that **expire after 7 days**, and the long-lived-token exception only
      covers basic profile scopes — not Gmail. So personal use means
      re-authorising weekly. Escaping that needs Google app verification, which
      for a Gmail scope means a security review.
- [ ] **`[me]`** `corpus/gmail.py` — OAuth flow, token cache, `messages.list` +
      batched `messages.get`, raw RFC822 → the same `Message` dataclass
- [ ] **`[me]`** Incremental sync via `historyId` so a re-run fetches only what
      is new instead of re-downloading the mailbox
- [ ] **`[me]`** Reuse the existing dedup + bulk filter unchanged, then
      **re-measure the bulk rate** — Enron's was 14.2%, modern Gmail will be far
      higher, and that number is worth reporting as a contrast
- [ ] **`[compute]`** Build one index with the config that won the ablation ·
      **~5–100 min depending on mailbox size**
- [ ] **`[me]`** Commitment pre-filter (date/deadline regex) + extraction scoped
      to a `--since` window
- [ ] **`[compute]`** Extract commitments over the recent window · **overnight**
- [x] **`[me]`** Local CLI or small local web UI for actual daily use — search
      and cited answer are built (`make serve`, `make ask`). Still missing:
      "what's due this week", which needs the router and extraction. The server
      binds to 127.0.0.1 and the bind address is deliberately not a flag.
- [ ] **`[me]`** Keep the private index out of git and off Hugging Face. The
      public demo stays Enron-only; `.gitignore` already covers `data/`, but
      verify before the first push.

---

## Separate track — higher value per hour than any of the above

The plan's own appendix says this, and it's right: **retrofitting numbers onto
your three existing projects will improve the resume more per hour than
finishing this project.** The models are already built; only the evaluation and
the write-up are missing.

- [ ] **`[you/me]`** Security Intelligence — AUC, FPR, flow count, CUSUM
      detection rate, p95 inference latency
- [ ] **`[you/me]`** Rainfall — ensemble F1 vs best single detector, over N
      years / M stations
- [ ] **`[you/me]`** KuChiKu — scoring rubric vs N human-rated sessions (κ),
      p50 voice round-trip latency
- [ ] **`[you]`** Decide how to handle Security Intelligence and Rainfall both
      reading as "anomaly detection + dashboard" — lead with the stronger, or
      reframe one around what makes it distinct

**Budget: one weekend.** Worth doing before Phase 4.
