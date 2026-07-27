"""Deciding whether a question is a search or a query.

This is the project's headline claim, and it is a claim about a *failure*: dense
retrieval collapses on temporal questions. "What's due next week" has no
semantically similar passage - the answer is a date comparison over extracted
commitments, and embedding the question retrieves messages that talk about
deadlines rather than messages with deadlines in the window. A per-class metric
makes that visible; a corpus-wide average hides it completely.

So questions are routed:

    sql       temporal and aggregate - answered by a WHERE clause over commitments
    hybrid    semantic and entity - answered by retrieval
    both      ambiguous - run both arms and merge, because a wrong route is worse
              than a slow answer

**Rules first, model second.** A regex that recognises "what's due this week" is
free, deterministic, and reproducible; an LLM classifier costs a round-trip per
query and can route the same question differently on two runs. The rules handle
the unambiguous cases and abstain otherwise, and only the abstentions reach the
model. This makes router accuracy decomposable - how much of it is the rules, how
much the model - which a single end-to-end number would not be.

**Abstention routes to `both`, never to a guess.** The cost of routing a temporal
question to retrieval is a wrong answer; the cost of `both` is one extra retrieval
pass. Those are not comparable, so uncertainty resolves toward the expensive side.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ROUTES = ("sql", "hybrid", "both")

# Confidence below which a rule-based decision is not trusted on its own.
RULE_CONFIDENCE = 0.9
MODEL_CONFIDENCE = 0.7
ABSTAIN_CONFIDENCE = 0.0


@dataclass(slots=True)
class Route:
    name: str
    reason: str = ""
    confidence: float = 0.0
    decided_by: str = "rules"

    def __post_init__(self) -> None:
        if self.name not in ROUTES:
            raise ValueError(f"unknown route {self.name!r}; have {list(ROUTES)}")


@dataclass(slots=True)
class RouterDecision:
    query: str
    route: str
    reason: str
    confidence: float
    decided_by: str
    matched: list[str] = field(default_factory=list)
    llm_calls: int = 0

    @property
    def uses_sql(self) -> bool:
        return self.route in ("sql", "both")

    @property
    def uses_retrieval(self) -> bool:
        return self.route in ("hybrid", "both")


# -- rules ------------------------------------------------------------------

# Phrases that make a question a date comparison rather than a similarity search.
# A question asking *what is due* in a window cannot be answered by finding
# messages that resemble it.
_TEMPORAL = re.compile(
    r"\b(?:"
    r"what(?:'s| is| are)?\s+(?:due|outstanding|pending|overdue|open)"
    r"|due (?:this|next|last|by|before|on|in)\b"
    r"|(?:this|next|last) (?:week|month|quarter|monday|tuesday|wednesday|thursday|friday)"
    r"|overdue|deadlines?|by when|how soon"
    r"|before (?:the )?end of (?:the )?(?:week|month|quarter|day)"
    r"|in the (?:next|last|past) \d+ (?:days?|weeks?|months?)"
    r"|between .{0,20}and .{0,20}\b(?:19|20)\d{2}"
    r"|\bon \d{4}-\d{2}-\d{2}\b"
    r")",
    re.IGNORECASE)

# Counting and ranking. "How many" is not a retrieval question at all: retrieval
# returns a ranked list, and the top of a ranked list is not a count.
_AGGREGATE = re.compile(
    r"\b(?:how many|how much|count of|number of|total (?:number|count|amount)"
    r"|list all|show all|every (?:commitment|deadline|message|email)"
    r"|most|fewest|busiest|average|per (?:week|month|person|sender))\b",
    re.IGNORECASE)

# Signals that the question wants content, not a date filter. Present alongside a
# temporal phrase, they make the question genuinely both.
_SEMANTIC = re.compile(
    r"\b(?:what did|what was (?:decided|agreed|said)|why|how come|explain"
    r"|about|regarding|discussion|context|reason|terms|details"
    r"|who (?:said|wrote|asked|decided|thinks?))\b",
    re.IGNORECASE)

_ENTITY = re.compile(
    r"\b(?:from|to|sent by|cc'?d|between)\s+[a-z][\w.\-]*(?:@|\s+[a-z])",
    re.IGNORECASE)


def classify_rules(query: str) -> Route | None:
    """A route, or None to abstain.

    Abstention is a deliberate outcome, not a failure: the rules cover the cases
    where the phrasing settles it and hand the rest to the model.
    """
    if not query or not query.strip():
        return None

    temporal = bool(_TEMPORAL.search(query))
    aggregate = bool(_AGGREGATE.search(query))
    semantic = bool(_SEMANTIC.search(query))
    entity = bool(_ENTITY.search(query))

    # An aggregate is never retrieval, whatever else it contains. A ranked list
    # cannot answer "how many".
    if aggregate:
        return Route("sql", "aggregate phrasing - a ranked list cannot be counted",
                     RULE_CONFIDENCE)

    # Temporal *and* content-seeking: "what did we agree is due next week" needs
    # the window from SQL and the substance from retrieval.
    if temporal and (semantic or entity):
        return Route("both", "temporal window plus a content question",
                     RULE_CONFIDENCE)

    if temporal:
        return Route("sql", "date window with no content question",
                     RULE_CONFIDENCE)

    if semantic or entity:
        return Route("hybrid", "content or entity question with no date window",
                     RULE_CONFIDENCE)

    return None


# -- the model fallback -----------------------------------------------------

SYSTEM = (
    "You classify questions about an email archive by how they should be "
    "answered. You return JSON and nothing else."
)

PROMPT = """\
An email archive can be queried two ways:

  sql     a date/count filter over extracted commitments - answers "what is due
          in this window", "how many", "list all". There is no text similarity
          involved.
  hybrid  semantic search over message text - answers "what was decided about X",
          "who handled Y", anything about content.
  both    the question needs a date window AND content, or you cannot tell.

Question: {query}

Return JSON only: {{"route": "sql|hybrid|both", "reason": "one short clause"}}

If it is not clear, answer "both". Routing a date question to search gives a
wrong answer; answering both ways is merely slower."""


class QueryRouter:
    """Rules first, model on abstention."""

    def __init__(self, llm=None, use_llm: bool = True) -> None:
        self._llm = llm
        self.use_llm = use_llm
        self.counts = {"rules": 0, "llm": 0, "default": 0}

    @property
    def llm(self):
        if self._llm is None:
            from ..llm.client import LLM
            self._llm = LLM()
        return self._llm

    def route(self, query: str) -> RouterDecision:
        decision = classify_rules(query)
        if decision is not None:
            self.counts["rules"] += 1
            return RouterDecision(query=query, route=decision.name,
                                  reason=decision.reason,
                                  confidence=decision.confidence,
                                  decided_by="rules")

        if not self.use_llm:
            self.counts["default"] += 1
            return RouterDecision(
                query=query, route="both",
                reason="rules abstained and no model is configured",
                confidence=ABSTAIN_CONFIDENCE, decided_by="default")

        try:
            data = self.llm.json_complete(PROMPT.format(query=query),
                                          system=SYSTEM, max_tokens=200)
            name = str((data or {}).get("route", "")).strip().lower()
            reason = str((data or {}).get("reason", ""))[:200]
            if name not in ROUTES:
                raise ValueError(f"model returned route {name!r}")
            self.counts["llm"] += 1
            return RouterDecision(query=query, route=name, reason=reason,
                                  confidence=MODEL_CONFIDENCE, decided_by="llm",
                                  llm_calls=1)
        except Exception as exc:                        # noqa: BLE001
            # A router failure must not fail the query. `both` is the safe
            # default for the same reason abstention is.
            self.counts["default"] += 1
            return RouterDecision(
                query=query, route="both",
                reason=f"router failed ({type(exc).__name__}), answering both ways",
                confidence=ABSTAIN_CONFIDENCE, decided_by="default", llm_calls=1)

    def render_counts(self) -> str:
        total = sum(self.counts.values()) or 1
        return " ".join(f"{k}={v} ({v/total:.0%})" for k, v in self.counts.items())


# -- accuracy ---------------------------------------------------------------

# How an eval-set query class maps to the route that answers it. This is the gold
# standard for router accuracy, and it is derived from the eval set's own
# stratification rather than hand-assigned per query.
CLASS_TO_ROUTE = {
    "temporal": "sql",
    "semantic": "hybrid",
    "entity": "hybrid",
    # An unanswerable control has no right route - it should produce no answer
    # whichever arm runs - so it is excluded from router accuracy rather than
    # counted as a hybrid win.
    "unanswerable": None,
}


def score_routes(decisions: list[RouterDecision], query_classes: dict[str, str]
                 ) -> dict:
    """Router accuracy against the eval set's query classes.

    `both` is scored as correct when it includes the right arm: the router's job is
    not to be minimal, it is to not miss. A separate `over_routed` count reports how
    often it paid for an arm it did not need, which is the real cost of that policy.
    """
    total = correct = over = 0
    by_class: dict[str, dict] = {}

    for d in decisions:
        cls = query_classes.get(d.query)
        want = CLASS_TO_ROUTE.get(cls)
        if want is None:
            continue
        total += 1
        bucket = by_class.setdefault(cls, {"n": 0, "correct": 0, "over": 0})
        bucket["n"] += 1

        hit = d.route == want or d.route == "both"
        if hit:
            correct += 1
            bucket["correct"] += 1
        if d.route == "both" and want != "both":
            over += 1
            bucket["over"] += 1

    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "over_routed": over,
        "over_routed_rate": over / total if total else 0.0,
        "by_class": {c: {**v, "accuracy": v["correct"] / v["n"] if v["n"] else 0.0}
                     for c, v in sorted(by_class.items())},
    }


def routes_table(score: dict) -> str:
    lines = ["| Query class | n | routed correctly | over-routed to `both` |",
             "|---|---|---|---|"]
    for cls, v in score["by_class"].items():
        lines.append(f"| {cls} | {v['n']} | {v['accuracy']:.3f} | {v['over']} |")
    lines.append(f"| **overall** | {score['n']} | {score['accuracy']:.3f} "
                 f"| {score['over_routed']} |")
    lines.append("")
    lines.append("`both` counts as correct when it includes the right arm - the "
                 "router's job is not to be minimal, it is to not miss. The "
                 "over-routed column is what that policy costs.")
    return "\n".join(lines)
