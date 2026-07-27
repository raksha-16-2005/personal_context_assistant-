"""Dimension 5: rewriting the query before it reaches the index.

Three arms, all of which spend an LLM call to buy retrieval quality:

*HyDE.* Embed a hypothetical *answer* rather than the question. A question and
the passage that answers it are not neighbours in embedding space - "who owns
the Calpine renewal" and a mail saying "I've picked up Calpine, closing Friday"
share almost no vocabulary. Generating the answer first moves the query vector
into the region the answer actually occupies.

*Multi-query expansion.* One question, k paraphrases, retrieve each, fuse. Buys
robustness to vocabulary mismatch: if any one phrasing lands on the corpus's
wording, RRF surfaces it.

*Decomposition.* Split a multi-hop question into sub-questions and retrieve each
separately. "What did legal say about the MSA before the steering committee
meeting" is two lookups, and no single chunk contains both.

Design points:

**One call, k outputs, temperature 0.** Multi-query wants k *different*
phrasings, and the tempting way to get them is k samples at temperature > 0.
That would be unreproducible and cost k calls against a free-tier quota. Asking
for a JSON array of k rewrites in one deterministic call is cheaper and
reproducible - and it is what makes the response cache effective, since the
whole expansion is one cache entry.

**HyDE writes an email, not an essay.** Indexed chunks begin with the literal
header block that `chunking._header` prepends ("from : ... to : ... subject :
..."). A hypothetical document shaped like a Wikipedia paragraph is stylistically
further from every chunk in the corpus than one shaped like a reply in a thread,
and dense retrieval scores style as readily as content.

**Degradation is counted, never silent.** A malformed response falls back to the
original query and increments `stats.degraded`. Dimension 5 has to report how
often its transform actually fired: "HyDE, 4 of 70 queries fell back" is a
result. A quiet fallback would publish the baseline's numbers under the
transform's name.

**Transforms are asked to fail loudly on unanswerable queries too.** The eval
set has 10 unanswerable controls, and a generative rewrite is exactly the step
that could invent plausible context for them. HyDE will happily hallucinate a
mail that does not exist; that is inherent to the method and is why refusal is
measured downstream in generation, not here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm.client import LLM, LLMError

# How many rewrites multi-query asks for. 3 is the usual setting and keeps the
# fused run interpretable; each extra rewrite is another retrieval pass, so this
# is also the latency knob.
N_REWRITES = 3

MAX_SUBQUESTIONS = 3


@dataclass(slots=True)
class TransformedQuery:
    """What the retriever should actually search for.

    `dense_texts` and `sparse_texts` are separate because the two retrievers
    want different things from a transform. HyDE's hypothetical document helps
    the dense side (it moves the vector) and can hurt BM25 (it injects invented
    terms that match nothing, diluting the real ones), so HyDE keeps the
    original text on the sparse side. Applying one rewrite to both retrievers
    would confound "does HyDE help" with "does HyDE help BM25", and the answers
    are different.
    """

    original: str
    dense_texts: list[str]
    sparse_texts: list[str]
    kind: str = "none"
    llm_calls: int = 0
    degraded: bool = False
    raw: str = ""

    @property
    def n_runs(self) -> int:
        """Retrieval passes this transform implies - the latency multiplier."""
        return max(len(self.dense_texts), len(self.sparse_texts))


def identity(query: str) -> TransformedQuery:
    return TransformedQuery(original=query, dense_texts=[query],
                            sparse_texts=[query], kind="none")


@dataclass(slots=True)
class TransformStats:
    calls: int = 0
    degraded: int = 0
    queries: int = 0
    fired: int = 0          # transforms that changed the query at all

    def render(self) -> str:
        return (f"{self.fired}/{self.queries} fired, "
                f"{self.degraded} degraded, {self.calls} llm calls")


HYDE_SYSTEM = (
    "You write a single plausible internal company email that would answer the "
    "user's question. You are not answering the question yourself - you are "
    "producing the kind of message that would contain the answer, so specific "
    "names, numbers and dates are welcome even if invented."
)

HYDE_PROMPT = """\
Question: {query}
{as_of_line}
Write one short email (4-6 sentences) from a colleague that would answer it.
Start with these exact header lines, filled in plausibly:

From: <sender@company.com>
To: <recipient@company.com>
Date: <YYYY-MM-DD>
Subject: <subject line>

Then the body. Output the email only - no commentary, no markdown fences."""

REWRITE_SYSTEM = (
    "You rewrite search queries over a corporate email archive. You return JSON "
    "and nothing else."
)

REWRITE_PROMPT = """\
Rewrite this search query {n} different ways for a keyword-and-semantic search \
over an email archive. Vary the vocabulary - use the words a colleague would \
actually have typed in a message about this, including likely synonyms, job \
titles and abbreviations. Keep every rewrite a standalone query.

Query: {query}

Return JSON only: {{"rewrites": ["...", "...", "..."]}}"""

DECOMPOSE_SYSTEM = (
    "You decompose search queries over an email archive into independent "
    "lookups. You return JSON and nothing else."
)

DECOMPOSE_PROMPT = """\
Does answering this query require finding more than one separate piece of \
information in an email archive? If it is a single lookup, say so - do not \
invent sub-questions.

Query: {query}

Return JSON only:
{{"multi_hop": true|false, "sub_questions": ["...", "..."]}}

At most {max_sub} sub-questions, each answerable by finding one message."""


class QueryTransformer:
    """Applies one dimension-5 arm, with per-run accounting.

    The LLM is shared and cached (see llm/cache.py): a second `make bench` over
    the same eval set makes zero network calls, which is the only reason
    dimension 5 can be re-run on a free-tier quota.
    """

    KINDS = ("none", "hyde", "multi_query", "decompose")

    def __init__(self, kind: str = "none", llm: LLM | None = None,
                 n_rewrites: int = N_REWRITES) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown transform {kind!r}; have {list(self.KINDS)}")
        self.kind = kind
        self.n_rewrites = n_rewrites
        # No client is constructed for the baseline arm, so `--transform none`
        # runs with no API key at all.
        self._llm = llm
        self.stats = TransformStats()
        self.log: list[dict] = []

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    def __call__(self, query: str, as_of: str = "") -> TransformedQuery:
        self.stats.queries += 1
        if self.kind == "none":
            return identity(query)

        fn = {"hyde": self._hyde, "multi_query": self._multi_query,
              "decompose": self._decompose}[self.kind]
        try:
            out = fn(query, as_of)
        except (LLMError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # A three-hour benchmark must not die on one malformed rewrite, but
            # the fallback is recorded so the table can report it.
            self.stats.degraded += 1
            out = identity(query)
            out.kind, out.degraded, out.raw = self.kind, True, f"{type(exc).__name__}: {exc}"

        self.stats.calls += out.llm_calls
        if not out.degraded and (out.dense_texts != [query] or out.sparse_texts != [query]):
            self.stats.fired += 1
        self.log.append({"query": query, "kind": out.kind, "degraded": out.degraded,
                         "n_runs": out.n_runs, "raw": out.raw[:2000]})
        return out

    # -- arms --------------------------------------------------------------

    def _hyde(self, query: str, as_of: str) -> TransformedQuery:
        as_of_line = (f"The question is being asked on {as_of}. Date the email "
                      f"plausibly relative to that.\n" if as_of else "")
        doc = self.llm.complete(
            HYDE_PROMPT.format(query=query, as_of_line=as_of_line),
            system=HYDE_SYSTEM, max_tokens=400).strip()
        if not doc:
            raise LLMError("hyde produced an empty document")

        # Dense side gets the hypothetical email; sparse side keeps the original.
        # An invented "From: sara.chen@company.com" is a gift to a bi-encoder and
        # pure noise to BM25, which would match the invented name literally.
        return TransformedQuery(original=query, dense_texts=[doc],
                                sparse_texts=[query], kind="hyde",
                                llm_calls=1, raw=doc)

    def _multi_query(self, query: str, as_of: str) -> TransformedQuery:
        data = self.llm.json_complete(
            REWRITE_PROMPT.format(query=query, n=self.n_rewrites),
            system=REWRITE_SYSTEM, max_tokens=512)
        rewrites = [r.strip() for r in _string_list(data, "rewrites") if r.strip()]
        if not rewrites:
            raise LLMError("multi_query returned no usable rewrites")

        # The original always stays in the run set. A paraphrase can be worse
        # than what the user typed, and fusion should never be able to lose a
        # result the baseline would have found.
        texts = [query] + rewrites[:self.n_rewrites]
        return TransformedQuery(original=query, dense_texts=texts,
                                sparse_texts=texts, kind="multi_query",
                                llm_calls=1, raw=json.dumps(rewrites))

    def _decompose(self, query: str, as_of: str) -> TransformedQuery:
        data = self.llm.json_complete(
            DECOMPOSE_PROMPT.format(query=query, max_sub=MAX_SUBQUESTIONS),
            system=DECOMPOSE_SYSTEM, max_tokens=512)
        if not isinstance(data, dict):
            raise LLMError(f"decompose returned {type(data).__name__}, not an object")

        subs = [s.strip() for s in _string_list(data, "sub_questions") if s.strip()]
        multi_hop = bool(data.get("multi_hop")) and len(subs) > 1
        if not multi_hop:
            # Not a degradation: "this query is one lookup" is the correct
            # answer for most of the eval set, and how often it fires is the
            # interesting number in the dimension-5 table.
            out = identity(query)
            out.kind, out.llm_calls = "decompose", 1
            out.raw = json.dumps({"multi_hop": False})
            return out

        texts = [query] + subs[:MAX_SUBQUESTIONS]
        return TransformedQuery(original=query, dense_texts=texts,
                                sparse_texts=texts, kind="decompose",
                                llm_calls=1, raw=json.dumps(subs))


def _string_list(data: object, key: str) -> list[str]:
    """Pull a list of strings out of a model response, tolerating the two
    shapes models actually return: the documented object, or a bare array."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get(key) or []
    else:
        raise LLMError(f"expected an object or array, got {type(data).__name__}")
    if not isinstance(items, list):
        raise LLMError(f"{key!r} is {type(items).__name__}, not a list")
    return [str(i) for i in items if isinstance(i, (str, int, float))]
