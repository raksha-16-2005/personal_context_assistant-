# Corpus construction, and three things the plan got wrong about it

Reproduce with `make corpus && make sample`.

```
517,401  files in the CMU maildir
-261,903  exact duplicate copies            (50.6%)
─────────
255,498  distinct messages
 -36,252  per-message bulk rules            (14.2%)
  -4,964  recurring (sender, subject) blasts (77 pairs)
─────────
214,282  indexable messages
 50,065  stratified ablation sample (34,700 whole threads)
```

## 1. Message-ID is useless for deduplication here

The obvious dedup key removes **exactly zero** messages. Enron's maildir was
exported through JavaMail, which stamped a fresh `Message-ID` on every copy —
so the sender's `sent` copy and each recipient's filed copy carry different
ids despite being the same message. All 435,498 rows of a first pass had
distinct ids.

Dedup is therefore on a content hash of
`(sender, recipients, cc, subject, date, body)`, which collapses **50.6%** of
the corpus. Recipients are inside the hash deliberately: copies of one message
share them, two separate sends of the same text to different people do not,
and entity-scoped queries depend on that distinction.

This is not a cosmetic saving. Left unfixed, the same message sits in the
index under several keys, so a query's gold set would have to enumerate every
copy or recall would silently under-report — and half the embedding budget
would be spent re-encoding text already in the index.

## 2. "~80% of mail is bulk" does not hold for a 1999–2002 corpus

That figure came from modern Gmail. Measured here, the per-message rules drop
**14.2%**. Dedup does the real reduction, not the bulk filter.

## 3. A per-message filter cannot see a broadcast

The rules in `filters.is_bulk` look at one message and cannot tell that the
same sender emitted the identical subject 623 times. Those blasts survived the
first build and did real damage downstream — see below. A corpus-level pass
drops any `(sender, subject_norm)` pair recurring **≥ 25** times: 4,964
messages across 77 pairs, all of them automated alerts, newsletters, and
mailbox-quota warnings.

## Threading: no reply headers exist

`linked_by_header = 0`. Not "few" — **zero**. The export preserved
`Message-ID` but dropped `In-Reply-To` and `References` entirely, so a
conventional references-based threader produces nothing but singletons and the
thread-aware chunking arm would be testing an empty condition.

All threading therefore comes from the heuristic pass: same normalized
subject **and** ≥ 1 shared participant **and** within 90 days. **This is a
caveat, not a detail** — thread-aware chunking results on this corpus rest on
reconstructed threads, and any reader should be told so.

### The over-merge, and the cap that fixes it

The first threading run produced a **1,124-message "thread"**. It was a
petition: 1,116 different people mailing Ken Lay the same subject line. Every
consecutive pair shares the recipient and falls inside the window, so
union-find chained all of them together. The next six offenders were automated
`schedule crawler: hourahead failure` alerts and newsletters.

Two fixes, and both were needed:

- the corpus-level blast filter above removed the automated senders
- pass 2 now refuses any subject bucket larger than **50** messages

Genuine threads on this corpus have a median size of 1 and a p99 of 7, so a
50-message bucket is a broadcast pattern rather than a conversation. Those are
left as singletons instead of being silently fused.

| | before | after |
|---|---|---|
| largest thread | 1,124 | 40 |
| oversized buckets skipped | — | 43 |
| multi-message threads | 83,101 | 33,446 |

## Sampling

50,065 messages across 34,700 **whole** threads — a thread is never split
across the sample boundary, or thread-aware chunking and multi-hop queries
would break for reasons unrelated to retrieval quality.

Strata are (year × thread-size bucket), with quotas proportional to each
stratum's share of *messages* rather than of threads. Sampling threads
uniformly would over-represent singletons, which are most of the thread count
but a minority of the messages.

The sample is drawn **before** the eval set is written, and queries are
authored against it. The reverse order means discovering that gold messages
for some queries fall outside the sample, which either caps recall below 1.0
or forces the sample to be patched around the labels.
