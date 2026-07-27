"""Dimension 6: why each recall@20 miss missed.

An aggregate recall number says how often retrieval fails. It says nothing about
*what* to fix, and the two are not the same question - a system at recall 0.72
whose misses are all vocabulary mismatch needs a different fix from one whose
misses are all temporal. This module classifies every miss into one of five
categories, and the counts become the README's "Where this fails" section.

    vocabulary    the answer exists but shares almost no terms with the query
    chunk_boundary  the message *was* retrieved, just not the chunk that answers
    temporal      the query is anchored in time and retrieval has no notion of it
    multi_hop     the answer is spread across messages, none of which is enough
    bad_label     the labelled message does not actually answer the query
    ranking       the terms *were* there and the retriever still ranked it low

The plan specifies the first five. `ranking` was added because the first five
cannot express the most common residual: a miss where the labelled message shares
most of the query's terms and was still ranked outside the cutoff. Filing that
under "vocabulary mismatch" would point the fix at query expansion when the
actual fix is reranking or fusion weights - the terms were already there. The
split is decided by measured term overlap, not by taste.

The categories are ordered by how cheaply they can be established, and each is
decided by the cheapest evidence that settles it:

**bad_label is checked first, and it is checked by a human.** Every other
category is a statement about the retriever; this one is a statement about the
eval set. Auto-classifying a miss as a bad label would let the system grade its
own exam - any query it fails could be dismissed as mislabelled. So this
category is only ever set by hand, in `notes`, and this module reports what a
human marked rather than deciding it.

**chunk_boundary is decided by arithmetic, not judgement.** A miss where some
chunk of the labelled message ranked well but the *answering* chunk did not is a
chunking failure, and the ranking already contains the evidence: look at where
the message's other chunks landed.

**temporal is decided by the query class.** The eval set already labels it, and
`as_of` marks queries whose answer depends on a date the index does not model.

**multi_hop is decided by how many messages are labelled.** A query with three
labelled messages where only one was found is a partial answer, not a total
failure, and it is a different fix (decomposition) from a query with one labelled
message that was missed entirely.

**vocabulary is the residual**, measured rather than assumed: term overlap
between the query and the labelled message's text is computed, so "vocabulary
mismatch" is a number and not a shrug.

Nothing here re-runs retrieval. It consumes the `per_query` records
`run_ablation.py` already writes, which keeps the taxonomy reproducible from a
results file and means re-classifying costs no compute.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Categories, in the order they are tested. Order matters: a temporal query that
# is also a vocabulary mismatch is counted as temporal, because that is the fix
# that would actually address it.
CATEGORIES = ("bad_label", "chunk_boundary", "temporal", "multi_hop",
              "vocabulary", "ranking")

# Marker a human writes in an eval-set note to disown a label. Checked as a
# substring so "bad_label: this is about a different contract" works.
BAD_LABEL_MARKER = "bad_label"

_WORD = re.compile(r"[a-z0-9']+")

# Terms that overlap between any query and any email and therefore say nothing
# about vocabulary mismatch. Deliberately short - a full stopword list would
# start removing content words like "due" and "sent" that carry real signal here.
STOPWORDS = frozenset("""
a an and are as at be been by did do does for from had has have he her his i if
in into is it its me my of on or our she that the their them there they this to
was we were what when where which who why will with you your about
""".split())

# Below this Jaccard-style overlap, a miss is a vocabulary mismatch rather than a
# ranking failure: there was almost nothing lexical for either retriever to grab.
VOCAB_OVERLAP_THRESHOLD = 0.10


def terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def overlap(query: str, document: str) -> float:
    """Fraction of the query's content terms that appear in the document."""
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(document)) / len(q)


@dataclass(slots=True)
class Miss:
    query_id: str
    query: str
    query_class: str
    category: str
    n_relevant: int
    n_found: int
    best_rank: int | None          # best position any relevant message reached
    best_chunk_rank: int | None    # best position any chunk of one reached
    term_overlap: float
    detail: str = ""


@dataclass
class FailureReport:
    misses: list[Miss] = field(default_factory=list)
    n_queries: int = 0
    n_answerable: int = 0
    k: int = 20

    @property
    def counts(self) -> Counter:
        return Counter(m.category for m in self.misses)

    def render(self) -> str:
        lines = [f"| Category | n | share of misses | example |",
                 "|---|---|---|---|"]
        counts = self.counts
        total = sum(counts.values()) or 1
        for cat in CATEGORIES:
            n = counts.get(cat, 0)
            if not n:
                continue
            example = next(m.query for m in self.misses if m.category == cat)
            lines.append(f"| {cat} | {n} | {n/total:.0%} | {example[:60]} |")
        lines.append(f"\n{len(self.misses)} of {self.n_answerable} answerable "
                     f"queries miss at recall@{self.k}.")
        return "\n".join(lines)


def classify(
    per_query: list[dict],
    queries_by_id: dict,
    chunk_ranks: dict[str, list[tuple[str, int]]] | None = None,
    texts_by_message: dict[str, str] | None = None,
    k: int = 20,
) -> FailureReport:
    """Classify every recall@k miss in one ablation row.

    `per_query` is the list `run_ablation.py` writes. `chunk_ranks` maps a
    query_id to [(chunk_id, rank), ...] for the labelled messages' chunks - it is
    what distinguishes a chunk-boundary failure from a total miss, and without it
    that category is simply never assigned rather than guessed at.
    """
    report = FailureReport(k=k, n_queries=len(per_query))

    for record in per_query:
        q = queries_by_id.get(record["query_id"])
        if q is None or not q.relevant_message_ids:
            continue                                  # unanswerable control
        report.n_answerable += 1

        found_at = record.get("found_at")
        if found_at is not None and found_at <= k:
            continue                                  # not a miss

        report.misses.append(_classify_one(
            q, record, chunk_ranks, texts_by_message, k))

    return report


def _classify_one(q, record: dict, chunk_ranks, texts_by_message, k: int) -> Miss:
    found_at = record.get("found_at")
    ranks = (chunk_ranks or {}).get(q.query_id, [])
    best_chunk_rank = min((r for _, r in ranks), default=None)

    doc_text = " ".join((texts_by_message or {}).get(m, "")
                        for m in q.relevant_message_ids)
    term_overlap = overlap(q.query, doc_text) if doc_text else 0.0

    miss = Miss(
        query_id=q.query_id, query=q.query, query_class=q.query_class,
        category="vocabulary", n_relevant=len(q.relevant_message_ids),
        n_found=1 if found_at else 0, best_rank=found_at,
        best_chunk_rank=best_chunk_rank, term_overlap=round(term_overlap, 3),
    )

    # 1. Only a human disowns a label.
    if BAD_LABEL_MARKER in (q.notes or "").lower():
        miss.category = "bad_label"
        miss.detail = f"marked by hand: {q.notes[:120]}"
        return miss

    # 2. A chunk of the right message ranked well; the answering chunk did not.
    if best_chunk_rank is not None and best_chunk_rank <= k and (
            found_at is None or found_at > k):
        miss.category = "chunk_boundary"
        miss.detail = (f"a chunk of the labelled message reached rank "
                       f"{best_chunk_rank}, the message itself only "
                       f"{found_at or '>depth'}")
        return miss

    # 3. The query is anchored in time; nothing in dense or sparse retrieval
    #    models a date. This is the router's job, not the retriever's.
    if q.query_class == "temporal":
        miss.category = "temporal"
        miss.detail = f"temporal query, as_of={q.as_of or 'unset'}"
        return miss

    # 4. Several labelled messages, at most some of them found: the answer is
    #    spread out and no single chunk carries it.
    if len(q.relevant_message_ids) > 1:
        miss.category = "multi_hop"
        miss.detail = (f"{len(q.relevant_message_ids)} labelled messages, best "
                       f"rank {found_at or '>depth'}")
        return miss

    # 5. Residual, split by measured lexical signal rather than assumed.
    if not doc_text:
        miss.category = "vocabulary"
        miss.detail = ("no message text available - overlap unmeasured, so the "
                       "vocabulary/ranking split could not be made")
    elif term_overlap < VOCAB_OVERLAP_THRESHOLD:
        miss.category = "vocabulary"
        miss.detail = (f"only {term_overlap:.0%} of query terms appear in the "
                       f"labelled message - little for either retriever to grab")
    else:
        miss.category = "ranking"
        miss.detail = (f"{term_overlap:.0%} of query terms appear in the labelled "
                       f"message, ranked {miss.best_rank or '>depth'} anyway - "
                       f"expansion would not fix this, reordering might")
    return miss


def chunk_ranks_from_run(ranked_chunks: list[tuple[str, float]],
                         relevant_messages: set[str]) -> list[tuple[str, int]]:
    """Positions of every chunk belonging to a labelled message.

    Ranks are 1-based positions in the *chunk* ranking, before the collapse to
    messages - which is the only place the chunk-boundary signal survives.
    """
    out: list[tuple[str, int]] = []
    for rank, (chunk_id, _score) in enumerate(ranked_chunks, start=1):
        if chunk_id.rsplit(":", 1)[0] in relevant_messages:
            out.append((chunk_id, rank))
    return out
