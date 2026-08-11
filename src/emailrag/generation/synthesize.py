"""Answer synthesis with citations - the generation half of the system.

Everything upstream of this module stops at a ranked list. This is where a
ranking becomes an answer, and it is the step where a retrieval system starts
being able to lie: the model can answer fluently from nothing, attribute a claim
to a message that does not contain it, or answer a question the corpus cannot
support. So three properties are enforced mechanically rather than requested
politely in a prompt.

**Refusal is a sentinel, not prose.** The eval set contains ten deliberately
unanswerable controls, and refusal rate is a reported metric. Detecting refusal by
pattern-matching apologies ("I'm sorry, I couldn't find...") would be a
string-matching problem with no correct answer, and it would count a hedged real
answer as a refusal. So the model is told to emit exactly `INSUFFICIENT_CONTEXT`,
and refusal becomes a boolean instead of a guess.

**Citations are validated against the sources actually passed in.** A `[7]` in the
output when six sources were supplied is a fabricated citation, and it is exactly
the failure a reader cannot detect by eye. Every marker is checked, out-of-range
ones are reported in `invalid_citations`, and the answer carries the count rather
than quietly rendering a broken link.

**Sources are messages, not chunks.** A chunk is an artifact of indexing; a
citation has to point at something a person can open. Chunks belonging to one
message are merged into a single numbered source, in retrieval order, so [3] means
the third-best *message* and not the third-best fragment.

What this module deliberately does not do: score its own output. Groundedness,
citation accuracy and answer relevance are measured in `generation/judge.py`
against a separate model, because a generator grading itself is not a measurement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.client import LLM, LLMError

# Emitted verbatim by the model when the sources do not answer the question.
# Machine-checkable on purpose - see the module docstring.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

# How many messages go into the prompt. Beyond ~8 the context is mostly
# distractors: the reranked head is where the answer is, and a longer prompt
# measurably raises the rate at which the model cites a plausible-but-wrong
# source. It is also the cost knob.
DEFAULT_N_SOURCES = 6

# Per-source truncation. A full thread-aware chunk can be 512 tokens and six of
# them will not fit a small model's usable attention span alongside the
# instructions.
MAX_SOURCE_CHARS = 1400

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM = f"""\
You answer questions about a corporate email archive, using only the numbered \
email excerpts you are given.

Rules, in order of importance:

1. If the excerpts do not contain the answer, reply with exactly \
{INSUFFICIENT} and nothing else. Do not guess, do not reason from general \
knowledge, and do not answer from what is plausible for a company like this. An \
unanswerable question is a normal outcome, not a failure.
2. Cite every factual claim with the bracketed number of the excerpt it came \
from, like [2]. A sentence stating a fact with no citation is a defect. Cite \
several as [1][3] where a claim rests on more than one.
3. Never cite a number you were not given.
4. Be specific. Names, dates, amounts and decisions are the point; if the \
excerpts disagree with each other, say so and cite both.
5. Answer in at most four sentences. This is a search result, not a report."""

PROMPT = """\
Question: {question}
{owner_line}{as_of_line}{route_note_line}
Numbered email excerpts:

{sources}

Answer the question using only these excerpts, citing them by number. If they do \
not contain the answer, reply with exactly {sentinel}."""


@dataclass(slots=True)
class Citation:
    """One numbered source offered to the model."""

    n: int
    message_id: str
    sender: str = ""
    recipients: str = ""
    date: str = ""
    subject: str = ""
    text: str = ""
    score: float = 0.0
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        who = self.sender.split("@")[0] if self.sender else "unknown"
        return f"[{self.n}] {who}, {self.date or 'undated'}: {self.subject or '(no subject)'}"


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    citations: list[Citation]
    cited_numbers: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    error: str = ""

    @property
    def cited_sources(self) -> list[Citation]:
        """Only the sources the answer actually used - what a UI should show
        prominently, with the rest available but folded away."""
        used = set(self.cited_numbers)
        return [c for c in self.citations if c.n in used]

    @property
    def is_grounded_shape(self) -> bool:
        """Cheap structural check: cited something, invented nothing.

        Not a groundedness *measurement* - that needs a judge reading the sources
        (see generation/judge.py). This only rules out the two failures visible
        without reading anything.
        """
        return (self.refused
                or (bool(self.cited_numbers) and not self.invalid_citations))


def _clip(text: str, limit: int = MAX_SOURCE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def format_sources(citations: list[Citation]) -> str:
    """Render sources for the prompt.

    Headers are included because half the questions this system is for are about
    *who* said something and *when*. A body-only excerpt makes "who owns the
    Calpine renewal" unanswerable even when the answering message is right there.
    """
    blocks = []
    for c in citations:
        blocks.append(
            f"[{c.n}]\n"
            f"From: {c.sender or 'unknown'}\n"
            f"To: {c.recipients or 'unknown'}\n"
            f"Date: {c.date or 'unknown'}\n"
            f"Subject: {c.subject or '(none)'}\n"
            f"{_clip(c.text)}"
        )
    return "\n\n".join(blocks)


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid cited numbers in order of first use, invalid ones).

    Deduplicated but order-preserving: the order sources are first cited is how a
    reader will scan them, so it is the order a UI should list them in.
    """
    seen: list[int] = []
    invalid: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            if n not in seen:
                seen.append(n)
        elif n not in invalid:
            invalid.append(n)
    return seen, invalid


# Sentences *about* the sources rather than claims drawn from them. A partial
# refusal - "the excerpts do not say who approved it" - is the behaviour this
# system wants most, and flagging it as an uncited claim would train a reader to
# ignore the flag. Found in real output: asked for the capital of France, the
# model answered "Paris is a city in France [5]. The excerpts do not explicitly
# state that Paris is the capital of France." The second sentence is the model
# declining to use its own world knowledge, which is exactly right.
#
# The source noun alone is not enough. "A confident assertion with no source at
# all" contains "no source" and is an uncited claim, not a statement about the
# excerpts - so a reporting verb is required near the noun. That verb is what makes
# a sentence a claim *about the sources* rather than a claim that happens to use
# the word.
_SOURCE_NOUN = r"(?:excerpts?|sources?|emails?|messages?|context|provided \w+)"
_REPORT_VERB = (r"(?:mention|state|say|indicate|specif|includ|contain|show|name"
                r"|address|discuss|confirm|describ|reference)")

_META = re.compile(
    rf"\b(?:the|these|those|any|none of the|no)\s+{_SOURCE_NOUN}\b"
    rf"[^.]{{0,60}}?\b{_REPORT_VERB}"
    rf"|\bnot\s+(?:mentioned|stated|specified|found|included|available|clear)\b"
    rf"|\bdo(?:es)?\s+not\s+(?:say|state|mention|specify|indicate|explicitly)\b",
    re.IGNORECASE)


def uncited_claim_sentences(text: str) -> list[str]:
    """Sentences that assert something about the world and cite nothing.

    A heuristic, and reported as such: it is a flag for a reader, not a metric.
    Skipped are short fragments, questions, connectives ending in a colon, and
    statements about the sources themselves.
    """
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = raw.strip()
        if len(sentence) < 25 or sentence.endswith("?") or sentence.endswith(":"):
            continue
        if _CITATION.search(sentence) or _META.search(sentence):
            continue
        out.append(sentence)
    return out


class Synthesizer:
    """Turns retrieved messages into a cited answer."""

    def __init__(self, llm: LLM | None = None, n_sources: int = DEFAULT_N_SOURCES,
                 max_tokens: int = 700) -> None:
        self._llm = llm
        self.n_sources = n_sources
        self.max_tokens = max_tokens

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    def answer(self, question: str, citations: list[Citation], as_of: str = "",
               mailbox_owner: str = "", route_note: str = "") -> Answer:
        """Synthesize an answer over `citations` (already numbered and ranked).

        `mailbox_owner`, when given, is the email address whose mail this is -
        the CLI/eval harness never passes it (the Enron corpus has no single
        "the user" to be), so it defaults to empty and leaves their prompt
        byte-identical. The web app does pass it: without it, nothing in the
        prompt ever says whose mailbox is being searched, and a question like
        "how many emails did I get today" is unanswerable in principle - not a
        retrieval miss, but the model correctly refusing to guess which
        excerpt's "To:" address "I" refers to. See webapp/app/chat/routes.py.

        `route_note`, same opt-in reasoning: the true count behind a "how
        many" question, computed by the date/SQL arm before `citations` was
        truncated to a handful of examples - see `Pipeline.ask`'s own
        docstring for why counting the truncated sample instead undercounts.
        """
        import time

        sources = citations[:self.n_sources]
        if not sources:
            # Nothing retrieved is a refusal, and it must not cost an LLM call:
            # asking a model to answer from an empty context invites it to answer
            # from its own weights instead.
            return Answer(question=question, text=INSUFFICIENT, citations=[],
                          refused=True)

        owner_line = (f"\nYou are answering on behalf of the mailbox owner, "
                     f"{mailbox_owner}. Resolve \"I\", \"me\", and \"my\" in the "
                     f"question to refer to them.\n" if mailbox_owner else "")
        as_of_line = (f"\nThe question is being asked on {as_of}. Resolve relative "
                      f"dates in the excerpts against that.\n" if as_of else "")
        route_note_line = (
            f"\nThe mailbox's date index reports: {route_note}. Use this exact "
            f"figure when answering \"how many\" - do not count the numbered "
            f"excerpts below instead, they may be a truncated sample of a larger "
            f"total. State it as a plain fact with no bracketed citation, since "
            f"it comes from the index rather than from any numbered excerpt.\n"
            if route_note else "")
        prompt = PROMPT.format(question=question, owner_line=owner_line,
                               as_of_line=as_of_line, route_note_line=route_note_line,
                               sources=format_sources(sources), sentinel=INSUFFICIENT)

        t0 = time.perf_counter()
        try:
            raw = self.llm.complete(prompt, system=SYSTEM,
                                    max_tokens=self.max_tokens).strip()
        except LLMError as exc:
            return Answer(question=question, text="", citations=sources,
                          error=f"{type(exc).__name__}: {exc}")
        elapsed = (time.perf_counter() - t0) * 1000

        # A model returning genuinely nothing is rare but real - observed from
        # one Gemini quota-rotation fallback model (llm/client.py's
        # GEMINI_FALLBACKS) on an ambiguous prompt, not every model. Without
        # this, an empty completion sailed through as `refused=False` with
        # blank `text` - a silent non-answer presented as if it had
        # succeeded, rather than the same honest refusal a real
        # `INSUFFICIENT_CONTEXT` would have been.
        if not raw:
            return Answer(question=question, text=INSUFFICIENT, citations=[],
                          refused=True, model=getattr(self.llm, "last_model", ""))

        refused = raw.replace("*", "").strip().upper().startswith(INSUFFICIENT)
        cited, invalid = parse_citations(raw, len(sources))

        return Answer(
            question=question,
            text=raw,
            citations=sources,
            cited_numbers=cited,
            invalid_citations=invalid,
            uncited_sentences=[] if refused else uncited_claim_sentences(raw),
            refused=refused,
            model=getattr(self.llm, "last_model", ""),
            cached=getattr(self.llm, "last_cached", False),
            latency_ms=elapsed,
        )
