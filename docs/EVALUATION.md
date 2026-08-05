# Evaluation methodology

Why these metrics, why these query classes, and what the numbers do **not**
support. Every table in the README is reproducible with `make bench` from the
eval set published in `data/eval/`.

Results sections are empty until the corresponding stage runs. Empty is
honest; a placeholder number is not.

---

## 1. What is being measured

**nDCG@10 is the headline, not accuracy.** Accuracy assumes one correct answer.
Retrieval returns a ranked list against a query with a *set* of relevant
messages, and position matters — a relevant message at rank 1 is worth more
than the same message at rank 9. nDCG is rank-weighted and normalized by the
best achievable ranking, so it is comparable across queries with different
numbers of relevant messages.

Reported alongside it:

| Metric | What it answers |
|---|---|
| recall@5 | Did the answer make it into what a user actually reads? |
| recall@20 | Did the retriever find it at all, for a reranker to promote? |
| MRR | How far down is the *first* useful result? |
| p50 / p95 latency | What does this cost at query time? |

recall@20 exists specifically to separate two failure modes that a single
metric conflates: the retriever never surfaced the document (unfixable by
reranking) versus it surfaced it at rank 17 (exactly what reranking fixes).

## 2. Relevance is binary, and scored at message level

**Binary, not graded.** Graded judgements need a second annotator to be
credible. One person labelling alone cannot report inter-annotator agreement,
and a graded scale without it is decoration. Binary keeps nDCG interpretable:
gain 1 for a labelled message, 0 otherwise.

**Message level, not chunk level.** This is the single most consequential
choice here. The eval set labels which *messages* answer a query; a retriever
is credited once for surfacing a message however many of its chunks appear.

Scoring chunks instead would break the chunking ablation outright. A strategy
emitting four chunks per message would outscore one emitting a single chunk
purely by occupying more of the top-k — an artifact of granularity, not
retrieval quality. Dimension 1 would then measure chunk count and be reported
as chunking quality.

The mechanical consequence: retrieval runs **deep in chunks** (200) and scores
**shallow in messages** (5, 10, 20). Retrieving only 20 chunks can collapse to
far fewer than 20 distinct messages, which would make recall@20 measure chunk
clustering.

## 3. Unanswerable controls are excluded, not zeroed

10 of the 80 queries have no answer in the corpus. They are **dropped from
recall, MRR, and nDCG** rather than scored zero.

Scoring them zero would drag every mean down by 12.5% and make these tables
incomparable to any published IR baseline — a reader seeing nDCG@10 = 0.52
cannot tell whether the system is mediocre or whether an eighth of the queries
were unanswerable by construction. There is no correct document to retrieve,
so the metric is undefined, and the code returns `NaN` rather than `0.0` to
make that explicit.

They are measured — on **refusal rate** in the generation eval, which is what
they were built to test.

## 4. How the eval set was built

Queries are drafted by an LLM from real threads, then **every one is
hand-verified**. No label in the published set comes from a model. An LLM that
both writes the query and guesses its answers produces an eval set that
measures the drafting model's blind spots.

### Pooling, and its bias

Judging all 50,065 messages per query is impossible. Judging only the source
thread is worse than impossible — it would mark relevant exactly the thread a
query was drafted from and silently mark every other genuine answer
irrelevant, penalising any retriever good enough to find them.

So this uses **pooling**, the standard TREC construction: the union of BM25
top-k, dense top-k, and the full source thread is judged by hand, and anything
outside the pool counts as non-relevant.

**The bias, stated plainly:** a relevant message that no pooled retriever
surfaced is never judged, so absolute recall is optimistic for every system.
It is *comparably* optimistic across configs, which is what the ablation tables
compare. Absolute numbers here should not be read as an estimate of true
recall against the full corpus.

### Stratification

| Class | n | Why it is a separate class |
|---|---|---|
| semantic / topical | 35 | The case dense retrieval is supposed to win |
| temporal / aggregate | 25 | The case it should lose — motivates the router |
| entity-scoped | 10 | Tests whether sender/recipient survives into the index |
| unanswerable | 10 | Refusal, not retrieval |

The split is not cosmetic. The router result depends entirely on it: a single
corpus-wide average hides the temporal collapse completely, and the headline
claim is only visible in the per-class breakdown.

## 5. Corpus caveats that bound these numbers

Full detail in [CORPUS.md](CORPUS.md). Three things affect interpretation:

**All threading is reconstructed.** `linked_by_header = 0` — Enron's export
preserved `Message-ID` but dropped `In-Reply-To` and `References` entirely.
Every thread here comes from a heuristic (same normalized subject, ≥ 1 shared
participant, within 90 days). Thread-aware chunking results therefore rest on
inferred threads, and a corpus with real reply headers might behave
differently.

**Subject buckets over 50 messages are not threaded at all.** Without that cap
a 1,116-sender petition became one 1,124-message "thread". Genuine threads here
have a median size of 1 and p99 of 7. Broadcast patterns are left as singletons.

**Deduplication is content-based and therefore approximate at the margin.**
Message-ID removes nothing on this corpus (every copy got a fresh id), so
identity is a hash of sender/recipients/cc/subject/date/body. Two genuinely
distinct messages identical in all six fields would be merged. This is
vanishingly rare and strictly preferable to the 50.6% duplication the naive
key leaves behind.

## 6. Ablations use exact search

Dimension 2 compares embedding models, so search must not vary with them. HNSW
recall loss depends on dimensionality and on how clustered a model's embedding
space is — an approximate comparison of bge-base against MiniLM would measure
the model *and* the index while attributing the result to the model.

Ablation retrieval is therefore exhaustive. At 50k messages that is one BLAS
call, faster than the ANN index it replaces. pgvector/HNSW backs the served
system, where the approximation loss is measured *against* these exact numbers
and reported rather than absorbed.

## 7. Weighted fusion is reported untuned

The weighted-fusion row uses a fixed 0.5/0.5. A sweep exists but is a footnote:
with 80 queries there is no honest dev/test split, so the best weight from a
sweep is selected on the same queries it is scored on and is an optimistic
upper bound, not a result.

## 8. Reranking measures reordering, not truncation

Dimension 4 reranks the top *k* chunks and leaves everything below them in their
original order. Truncating at *k* would be the obvious implementation and would
corrupt the result: metrics are message-level over a deep chunk ranking, so
cutting the ranking at *k* chunks reduces the number of *distinct messages*
available to recall@20. A reranker that reordered the head perfectly could then
post a recall drop caused entirely by truncation, and the table would read as
"reranking hurts recall".

Two further decisions:

**The cross-encoder scores the user's query, never the transform's output.** When
dimensions 4 and 5 are combined, a HyDE document is a retrieval device — a
fabricated email — and scoring passages for similarity to a fabrication would
measure agreement with the generator rather than relevance to the question.

**Passages are the text the index holds, not the original message.** Chunking
round-trips text through the reference tokenizer, so an indexed chunk is
lowercased and detokenized. Scoring the pristine original would score a passage
the retriever never saw. Both candidate rerankers are uncased BERT-family, so the
cost is punctuation spacing.

### Passage length is an arm, not a detail

Cross-encoder cost grows superlinearly in sequence length, and
`make bench-rerank-budget` measures 128-token passages at 2–5× the throughput of
512-token ones on this CPU. Unlike lowering *k*, truncation does not shrink the
candidate set — every chunk still gets scored, just on less of itself. So the
default arms isolate the two levers separately:

| Arm | changes vs the one above |
|---|---|
| `L6@20` | the plan's original config |
| `L6@20/t192` | passage length only |
| `L2@20/t192` | model depth only |

Without both, a quality drop in the shipped arm could not be attributed to
either lever.

## 9. Query transformation reports how often it fired

A generative rewrite can fail — malformed JSON, an empty document, a spent quota
— and the safe behaviour is to fall back to the original query. That fallback is
counted, not silent. A transform that quietly degraded on a third of the eval set
would be reporting the baseline's numbers under its own name, and no amount of
nDCG makes that interpretable without the count. `fired` and `degraded` are
columns in the dimension-5 table.

Two structural decisions:

**Rewrites are fused within a modality before across it.** Four query rewrites
would otherwise outvote the sparse retriever 4:1 inside a "50/50" hybrid, making
a dimension-5 change silently a dimension-3 change too. So the rewrites are fused
by RRF first, then the dense and sparse sides combine by whatever dimension 3
specifies.

**HyDE rewrites the dense side only.** An invented sender address is useful to a
bi-encoder and pure noise to BM25, which would match the fabricated name
literally. One rewrite applied to both retrievers would conflate "does HyDE help"
with "does HyDE help BM25".

**k rewrites come from one call at temperature 0.** The obvious alternative — k
samples at temperature > 0 — is unreproducible and costs k times the quota.

## 10. Every LLM call is cached, and that is a methodological choice

Dimension 5, the router and the judge all call an LLM per query per config.
Without a cache a single `make bench` exceeds the free-tier daily quota, which
means the tables cannot be regenerated the same day — and reproducibility from
the published eval set is the project's central claim.

The cache is keyed on provider, model, temperature, max_tokens, system prompt and
user prompt. The model is part of the key deliberately: the extraction comparison
is local-model-versus-hosted-model, and serving one model's cached answer for the
other would fabricate that result. Lookups are rotation-aware, so a prompt
answered by a fallback model yesterday is not re-paid as a 429 today.

There is no TTL. Staleness is the point: a benchmark that quietly re-queries a
newer model version stops being reproducible.

## 11. What these numbers do not support

- **Absolute recall against the full 214,282-message corpus.** Ablations run on
  a 50,065-message stratified sample; pooling adds further optimism.
- **Generalisation beyond 1999–2002 corporate email.** Vocabulary, threading
  conventions, and bulk-mail patterns all differ from modern mail.
- **Latency on other hardware.** Every timing is CPU-only on one 2019 i7-9750H.
  See [HARDWARE.md](HARDWARE.md). This applies with particular force to
  dimension 4: the arm selection there is a consequence of *this* CPU's
  instruction set, and a machine with AVX-512 VNNI would likely reach a different
  conclusion about int8 quantization.
- **Statistical significance between close configurations.** 70 answerable
  queries is a small sample; differences under roughly 0.02 nDCG should be read
  as noise unless a significance test is reported alongside them.

## 12. Where quality and speed are allowed to come from different machines

Embedding may be computed on a GPU (see `notebooks/kaggle_build_indices.ipynb`)
while latency is measured locally. The rule:

> **Quality metrics (recall, MRR, nDCG) are hardware-independent** — same
> weights, same math, same answer. Compute them wherever is fastest.
> **Latency metrics are hardware-specific.** Measure them on one stated platform
> and never mix platforms within a table.

Stated explicitly this is more rigorous than hiding it; stated carelessly it is
dishonest. Mechanically: every imported index records `embed_platform` in its
`config.json`, so a T4 throughput figure cannot end up in a table of CPU numbers.
Retrieval latency is unaffected by where the vectors were built — searching the
matrix is local CPU work either way — so only the embed-throughput column is
platform-bound.

The import path refuses anything it cannot prove came from the same corpus: local
re-chunking must reproduce every chunk id, **in order**, since row *i* of the
matrix is chunk *i* and nothing else records that pairing. It also rejects
non-unit-length rows, because `DenseIndex` treats the dot product *as* the cosine.

---

## Results

**Pilot run: 26 queries (24 answerable + 2 unanswerable), hand-verified —
not yet the full 80-query set section 5's stratification targets.** Read
this as "the harness produces sane numbers," not as a final result. Per
section 11, differences under ~0.02 nDCG here should be read as noise; with
24 answerable queries that threshold is generous, not conservative.

### Retriever comparison

Pivot config: `thread_aware` / `all-MiniLM-L6-v2`.

| Configuration | R@5 | R@20 | MRR | nDCG@10 | p95 ms |
|---|---|---|---|---|---|
| BM25 only | 0.417 | 0.467 | 0.441 | 0.391 | 4 |
| Dense only | 0.454 | 0.572 | 0.535 | 0.491 | 30 |
| Hybrid — RRF | 0.386 | 0.624 | 0.345 | 0.365 | 32 |
| Hybrid — weighted (0.5/0.5, untuned per section 7) | 0.591 | 0.715 | 0.427 | 0.478 | 30 |

Weighted fusion wins on every chunking strategy tried (dimension 1), the one
consistent signal in this pilot.

### Reranking — nDCG gain against added latency

| Rerank arm | nDCG@10 | ΔnDCG | rerank p50 ms |
|---|---|---|---|
| none (baseline) | 0.365 | — | — |
| L-6 @ top-20, 512 tok | 0.348 | -0.018 | 628 |
| L-6 @ top-20, 192 tok | 0.347 | -0.018 | 381 |
| L-2 @ top-20, 192 tok | 0.326 | -0.039 | 158 |

Every arm lowers nDCG@10 on this pilot. Given the sample size this is not
read as "reranking fails here" — it is read as "re-run once the eval set is
bigger before concluding anything." Latency column is the previously
clean-measured baseline; this session's own rerank run shared the CPU with
`index-d2` building concurrently, so its raw latency readings are not used
here (quality metrics are unaffected by concurrent load, latency is).

### Query transformation — including the arms that lose

| Transform | fired | degraded | nDCG@10 | ΔnDCG | llm calls |
|---|---|---|---|---|---|
| none (baseline) | — | — | 0.365 | — | 0 |
| HyDE | 17/26 | 9 | 0.338 | -0.027 | 17 |
| Multi-query expansion | 26/26 | 0 | 0.319 | -0.046 | 26 |
| Decomposition | 7/26 | 18 | 0.370 | +0.005 | 8 |

Decompose is roughly flat and fired on only 7 of 26 queries (mostly
degrading to the baseline query); HyDE and multi-query both hurt. Per
section 9, `fired`/`degraded` are reported precisely so a quietly-degraded
transform cannot claim the baseline's numbers under its own name.

### Per-class breakdown

BM25, pivot config — the case section 4's stratification exists to isolate:

| Query class | n | R@5 | R@20 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| entity | 7/7 | 0.800 | 0.893 | 0.536 | 0.563 |
| semantic | 8/8 | 0.438 | 0.750 | 0.372 | 0.403 |
| temporal | 9/9 | 0.602 | 0.806 | 0.508 | 0.534 |
| unanswerable | 0/2 | — | — | — | — |

The stratification counts (8/9/7/2) are what's labelled so far, not the
plan's targets (35/25/10/10) — see the warning `make bench` prints.

### Failure taxonomy

6 of 24 answerable queries miss at recall@20 in this pilot:

| Category | n | share of misses |
|---|---|---|
| temporal | 1 | 17% |
| vocabulary | 5 | 83% |

No `chunk_boundary`, `multi_hop`, or `ranking` misses appeared at this
sample size. 5 of the 6 vocabulary misses report "no message text available"
for the term-overlap measurement — a consequence of thread-aware chunking
bundling multiple messages into one chunk under a single anchor dedup_key
(section on chunk storage in the README), not a defect in the labels
themselves. `bad_label` was not assigned to any query — see section 3's
rule that a system does not get to grade its own labels.

### HNSW recall loss (pgvector), pivot config

Section 6 explains why the ablation itself uses exact search. This is the
number that search decision was deferring — how much recall pgvector's HNSW
index actually gives up against it, at each `ef_search`:

| ef_search | recall@20 vs exact | worst query | top-1 agreement | p50 ms | p95 ms |
|---|---|---|---|---|---|
| 40 | 0.792 | 0.100 | 0.885 | 1.8 | 2.7 |
| 100 | 0.917 | 0.700 | 0.923 | 2.2 | 2.6 |
| 200 | 0.962 | 0.750 | 0.923 | 3.6 | 4.5 |
| 400 | 0.989 | 0.950 | 0.962 | 5.7 | 6.8 |

"Recall" here means agreement with exact search, not relevance — a chunk
exact search never surfaces cannot be recovered by an approximate index
either. `ef_search=40`'s worst-query recall (0.10) says the default is too
low for at least one query in a 26-query pilot; `ef_search=200` is the
first setting where the worst case clears 0.7. Latency is CPU-specific, see
HARDWARE.md.
