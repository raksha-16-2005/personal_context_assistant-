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

## 8. What these numbers do not support

- **Absolute recall against the full 214,282-message corpus.** Ablations run on
  a 50,065-message stratified sample; pooling adds further optimism.
- **Generalisation beyond 1999–2002 corporate email.** Vocabulary, threading
  conventions, and bulk-mail patterns all differ from modern mail.
- **Latency on other hardware.** Every timing is CPU-only on one 2019 i7-9750H.
  See [HARDWARE.md](HARDWARE.md).
- **Statistical significance between close configurations.** 70 answerable
  queries is a small sample; differences under roughly 0.02 nDCG should be read
  as noise unless a significance test is reported alongside them.

---

## Results

*Populated by `make bench`. Empty until the pilot runs.*

### Retriever comparison

### Per-class breakdown

### Failure taxonomy
