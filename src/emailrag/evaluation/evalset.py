"""The published eval set: schema, loading, and validation.

This file is the contract for `data/eval/queries.jsonl`. The eval set is the
one artifact of this project that is genuinely reusable by someone else, so it
is validated strictly rather than trusted - a silently malformed label is
worse than a missing one, because it produces a number that looks fine.

Validation is not optional decoration. `validate()` runs in CI and in
`make bench`; every table in the README is generated from a file that passed
it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Counts come from the plan's stratification. They are enforced as a *warning*
# rather than an error: the split is a design target, and a run against a
# partially-built eval set should still work while reporting the shortfall.
TARGET_COUNTS = {
    "semantic": 35,
    "temporal": 25,
    "entity": 10,
    "unanswerable": 10,
}
QUERY_CLASSES = frozenset(TARGET_COUNTS)


@dataclass(slots=True)
class EvalQuery:
    query_id: str
    query: str
    query_class: str
    relevant_message_ids: list[str]
    notes: str = ""
    verified: bool = False
    source_thread_id: str = ""
    # ISO date the query is asked *as of*. Required for temporal queries and
    # meaningless for the rest.
    #
    # "What's due this Thursday" - the plan's own example - has no answer over
    # a frozen 1999-2002 corpus, because "this Thursday" resolves against the
    # moment the question is asked and the benchmark has no such moment.
    # Anchoring each temporal query to a date makes it well-defined, and it is
    # the same anchor the extraction step needs to turn "next Thursday" in a
    # message into an absolute due_at.
    as_of: str = ""

    @property
    def answerable(self) -> bool:
        return self.query_class != "unanswerable"


class ValidationError(Exception):
    pass


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  warning {w}")
        return "\n".join(lines) or "  clean"


def load(path: Path) -> list[EvalQuery]:
    out: list[EvalQuery] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{path}:{lineno}: bad JSON - {exc}") from exc
        try:
            out.append(EvalQuery(
                query_id=raw["query_id"],
                query=raw["query"],
                query_class=raw["query_class"],
                relevant_message_ids=list(raw.get("relevant_message_ids", [])),
                notes=raw.get("notes", ""),
                verified=bool(raw.get("verified", False)),
                source_thread_id=raw.get("source_thread_id", ""),
                as_of=raw.get("as_of", ""),
            ))
        except KeyError as exc:
            raise ValidationError(f"{path}:{lineno}: missing field {exc}") from exc
    return out


def validate(queries: list[EvalQuery], corpus_ids: set[str] | None = None) -> ValidationReport:
    """Check the eval set for the failure modes that silently corrupt results.

    `corpus_ids` is the set of dedup_keys actually in the indexed corpus. Pass
    it whenever possible: a label pointing at a message outside the corpus caps
    recall below 1.0 for reasons that have nothing to do with the retriever,
    and it is completely invisible in the aggregate numbers.
    """
    report = ValidationReport()
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}

    for q in queries:
        where = f"{q.query_id}"

        if q.query_id in seen_ids:
            report.errors.append(f"{where}: duplicate query_id")
        seen_ids.add(q.query_id)

        if q.query_class not in QUERY_CLASSES:
            report.errors.append(
                f"{where}: unknown query_class {q.query_class!r} "
                f"(expected one of {sorted(QUERY_CLASSES)})")

        if not q.query.strip():
            report.errors.append(f"{where}: empty query text")

        normalized = " ".join(q.query.lower().split())
        if normalized in seen_text:
            report.warnings.append(
                f"{where}: near-duplicate of {seen_text[normalized]} - "
                f"inflates whichever class it lands in")
        seen_text[normalized] = q.query_id

        # The core invariant. An unanswerable control with labels is not a
        # control; an answerable query without labels scores 0 forever.
        if q.answerable and not q.relevant_message_ids:
            report.errors.append(f"{where}: answerable query has no relevant_message_ids")
        if not q.answerable and q.relevant_message_ids:
            report.errors.append(
                f"{where}: unanswerable control has {len(q.relevant_message_ids)} "
                f"labels - it must have none")

        if len(set(q.relevant_message_ids)) != len(q.relevant_message_ids):
            report.errors.append(f"{where}: duplicate ids in relevant_message_ids")

        if not q.verified:
            report.warnings.append(f"{where}: not hand-verified")

        # A temporal query with relative language and no anchor is unanswerable
        # by construction, and would be scored as a retrieval failure.
        if q.query_class == "temporal":
            if not q.as_of:
                report.warnings.append(
                    f"{where}: temporal query without as_of - relative phrasing "
                    f"has no referent over a frozen corpus")
            elif not _ISO_DATE.match(q.as_of):
                report.errors.append(f"{where}: as_of {q.as_of!r} is not YYYY-MM-DD")

        if corpus_ids is not None:
            missing = [m for m in q.relevant_message_ids if m not in corpus_ids]
            if missing:
                report.errors.append(
                    f"{where}: {len(missing)} labelled message(s) absent from the "
                    f"corpus, e.g. {missing[0]!r} - caps recall below 1.0")

    counts = {cls: sum(1 for q in queries if q.query_class == cls) for cls in TARGET_COUNTS}
    for cls, target in TARGET_COUNTS.items():
        if counts[cls] != target:
            report.warnings.append(
                f"class {cls}: {counts[cls]} queries, plan targets {target}")

    return report


def stratification(queries: list[EvalQuery]) -> dict[str, int]:
    out = {cls: 0 for cls in TARGET_COUNTS}
    for q in queries:
        out[q.query_class] = out.get(q.query_class, 0) + 1
    return out
