"""Date accuracy, and the local-versus-ceiling comparison.

The headline number for extraction is **date accuracy**, not commitment count. A
system that finds every commitment and dates a third of them wrong is worse than
one that finds fewer and dates them right, because a wrong due date is acted on.

Three ways of scoring a date, reported separately because they answer different
questions:

    exact          the resolved date equals the gold date
    within_1d      off by at most a day - a "Friday"/"end of week" boundary case
    either_reading an ambiguous phrase matched under one of its two conventions

`either_reading` exists because "next Thursday" has no single correct answer (see
dates.py). Scoring those strictly measures whether the annotator and the convention
agree, not whether the extractor understood the sentence. Reporting exact and
either_reading side by side makes the size of that ambiguity visible instead of
hiding it in one number.

For the model comparison, agreement is computed on the *messages both arms saw*,
never on each arm's own output - two arms that extracted different commitments have
no shared denominator, and a comparison over different denominators is not one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass(slots=True)
class DateScore:
    n: int = 0
    exact: int = 0
    within_1d: int = 0
    either_reading: int = 0
    ambiguous: int = 0
    missing: int = 0            # gold has a date, extraction produced none
    spurious: int = 0           # extraction produced a date, gold has none

    def rate(self, field_name: str) -> float:
        return getattr(self, field_name) / self.n if self.n else 0.0

    def render(self) -> str:
        return (
            f"| n | exact | within 1d | either reading | ambiguous | missing | spurious |\n"
            f"|---|---|---|---|---|---|---|\n"
            f"| {self.n} | {self.rate('exact'):.3f} | {self.rate('within_1d'):.3f} "
            f"| {self.rate('either_reading'):.3f} | {self.ambiguous} | {self.missing} "
            f"| {self.spurious} |"
        )


def score_dates(pairs: list[tuple[object, object]]) -> DateScore:
    """Score resolved dates against gold.

    `pairs` is [(commitment, gold), ...] where gold has `due_at` (a date or ISO
    string, or None). A commitment contributes to exactly one of exact /
    within_1d-only / neither, and `either_reading` counts separately because an
    ambiguous phrase can be right under a convention nobody picked.
    """
    score = DateScore()
    for commitment, gold in pairs:
        got = _as_date(getattr(commitment, "due_at", None))
        want = _as_date(getattr(gold, "due_at", None) if not isinstance(gold, dict)
                        else gold.get("due_at"))

        if want is None and got is None:
            continue
        if want is None:
            score.spurious += 1
            continue
        if got is None:
            score.missing += 1
            score.n += 1
            continue

        score.n += 1
        if getattr(commitment, "due_ambiguous", False):
            score.ambiguous += 1

        delta = abs((got - want).days)
        if delta == 0:
            score.exact += 1
        if delta <= 1:
            score.within_1d += 1

        alternative = _as_date(getattr(commitment, "due_alternative", None))
        if delta == 0 or (alternative is not None and alternative == want):
            score.either_reading += 1
    return score


@dataclass
class ArmComparison:
    """Two extraction arms over the same messages."""

    local_model: str = ""
    ceiling_model: str = ""
    messages: int = 0
    local_commitments: int = 0
    ceiling_commitments: int = 0
    both: int = 0                # commitments matched between the arms
    local_only: int = 0
    ceiling_only: int = 0
    local_dates: DateScore = field(default_factory=DateScore)
    ceiling_dates: DateScore = field(default_factory=DateScore)

    @property
    def agreement(self) -> float:
        """Jaccard over matched commitments. Not accuracy - neither arm is gold."""
        union = self.both + self.local_only + self.ceiling_only
        return self.both / union if union else 0.0

    def render(self) -> str:
        return "\n".join([
            f"| | {self.local_model} | {self.ceiling_model} |",
            "|---|---|---|",
            f"| commitments | {self.local_commitments} | {self.ceiling_commitments} |",
            f"| only this arm | {self.local_only} | {self.ceiling_only} |",
            f"| date exact | {self.local_dates.rate('exact'):.3f} "
            f"| {self.ceiling_dates.rate('exact'):.3f} |",
            f"| date either reading | {self.local_dates.rate('either_reading'):.3f} "
            f"| {self.ceiling_dates.rate('either_reading'):.3f} |",
            "",
            f"{self.both} commitments matched across arms over {self.messages} "
            f"messages (Jaccard {self.agreement:.3f}). Neither arm is gold, so this "
            f"is agreement, not accuracy.",
        ])


def _normalize(text: str) -> set[str]:
    return {w for w in "".join(
        c if c.isalnum() else " " for c in (text or "").lower()).split() if len(w) > 2}


def same_commitment(a, b, threshold: float = 0.5) -> bool:
    """Whether two extractions describe the same obligation.

    Token overlap on the text, within the same message. Fuzzy because two models
    will word the same commitment differently ("send the redline" vs "deliver
    redlined MSA"); requiring identical strings would report near-total
    disagreement between arms that agreed.
    """
    if getattr(a, "message_id", None) != getattr(b, "message_id", None):
        return False
    ta, tb = _normalize(getattr(a, "text", "")), _normalize(getattr(b, "text", ""))
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def compare_arms(local: list, ceiling: list, gold: dict | None = None,
                 local_model: str = "local", ceiling_model: str = "ceiling"
                 ) -> ArmComparison:
    """Match two arms' commitments and score both against gold where available."""
    comparison = ArmComparison(local_model=local_model, ceiling_model=ceiling_model)
    comparison.local_commitments = len(local)
    comparison.ceiling_commitments = len(ceiling)
    comparison.messages = len({c.message_id for c in local} |
                              {c.message_id for c in ceiling})

    unmatched = list(ceiling)
    for a in local:
        match = next((b for b in unmatched if same_commitment(a, b)), None)
        if match is not None:
            unmatched.remove(match)
            comparison.both += 1
        else:
            comparison.local_only += 1
    comparison.ceiling_only = len(unmatched)

    if gold:
        comparison.local_dates = score_dates(_pair_with_gold(local, gold))
        comparison.ceiling_dates = score_dates(_pair_with_gold(ceiling, gold))
    return comparison


def _pair_with_gold(commitments: list, gold: dict) -> list[tuple[object, object]]:
    """Pair each commitment with its gold row, matched within the message.

    Only commitments whose message has gold labels are scored. Scoring a
    commitment from an unlabelled message against a missing gold row would count
    every extraction on unlabelled data as spurious.
    """
    out = []
    for c in commitments:
        candidates = gold.get(c.message_id)
        if not candidates:
            continue
        match = next((g for g in candidates if same_commitment(c, g)), None)
        if match is not None:
            out.append((c, match))
    return out
