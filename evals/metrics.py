"""Metric definitions and percentile bootstrap CIs.

Every metric is a ratio of sums. Each ticket contributes a (numerator,
denominator) pair, and the metric is sum(num) / sum(den). Bootstrap resamples
whole tickets, which is the right unit here: a ticket is the independent
observation, not a tool call.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from evals.schema import AgentResult, Outcome, Ticket

Contribution = tuple[float, float]
MetricFn = Callable[[Ticket, AgentResult], Contribution]

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260824


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    numerator: float
    denominator: float
    undefined_resamples: int
    caveat: str

    @property
    def is_degenerate(self) -> bool:
        """CI collapsed to a point — the resamples carried no information."""
        return (
            self.ci_low is not None
            and self.ci_high is not None
            and math.isclose(self.ci_low, self.ci_high)
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "undefined_resamples": self.undefined_resamples,
            "degenerate_ci": self.is_degenerate,
            "caveat": self.caveat,
        }


def _resolution_rate(ticket: Ticket, result: AgentResult) -> Contribution:
    correct = (
        result.outcome is Outcome.RESOLVE
        and ticket.labels.outcome is Outcome.RESOLVE
    )
    return (1.0 if correct else 0.0, 1.0)


def _escalation_precision(ticket: Ticket, result: AgentResult) -> Contribution:
    if result.outcome is not Outcome.ESCALATE:
        return (0.0, 0.0)
    return (1.0 if ticket.labels.outcome is Outcome.ESCALATE else 0.0, 1.0)


# TODO(real-agent): name-level matching is deliberate for now, but argument
# correctness, call ordering, and repeat/loop detection each need to become
# their own metric once the agent lands. Folding them into this number would
# hide which of the four is failing; leaving them unmeasured means an agent
# that loops on a correct tool still scores 1.0 here.
def _tool_call_precision(ticket: Ticket, result: AgentResult) -> Contribution:
    expected = set(ticket.labels.expected_tools)
    hits = sum(1 for call in result.tool_calls if call.tool in expected)
    return (float(hits), float(len(result.tool_calls)))


def _pending_approval_rate(ticket: Ticket, result: AgentResult) -> Contribution:
    return (1.0 if result.outcome is Outcome.PENDING_APPROVAL else 0.0, 1.0)


METRICS: dict[str, tuple[MetricFn, str]] = {
    "resolution_rate": (
        _resolution_rate,
        "Denominator is the whole set, so this is capped by the share of "
        "tickets whose gold outcome is resolve. Resolving a ticket that should "
        "have been escalated scores zero, it is not merely uncounted.",
    ),
    "escalation_precision": (
        _escalation_precision,
        "Undefined when the agent escalates nothing. An agent that escalates "
        "everything scores exactly the gold escalate share; high precision "
        "here is worthless without resolution rate beside it.",
    ),
    "tool_call_precision": (
        _tool_call_precision,
        "Name-level match against the label's expected_tools. It ignores "
        "argument correctness, call order, and repeated calls, so a loop that "
        "calls a correct tool ten times scores 1.0. Recall is not measured.",
    ),
    "pending_approval_rate": (
        _pending_approval_rate,
        "A rate, not a precision: it says how often the agent asked for "
        "approval, not whether it was right to. Compare against the gold "
        "pending_approval share printed with the label distribution.",
    ),
}


def _ratio(pairs: list[Contribution]) -> float | None:
    denominator = sum(d for _, d in pairs)
    if denominator == 0:
        return None
    return sum(n for n, _ in pairs) / denominator


def _percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def bootstrap_ci(
    pairs: list[Contribution],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, int]:
    """Percentile bootstrap over tickets. Returns (low, high, undefined_count)."""
    if not pairs:
        return (None, None, resamples)

    rng = random.Random(seed)
    n = len(pairs)
    values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        value = _ratio(sample)
        if value is None:
            undefined += 1
        else:
            values.append(value)

    if not values:
        return (None, None, undefined)

    values.sort()
    return (
        _percentile(values, alpha / 2),
        _percentile(values, 1 - alpha / 2),
        undefined,
    )


def evaluate(
    tickets: list[Ticket],
    results: dict[str, AgentResult],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[MetricResult]:
    out: list[MetricResult] = []
    for name, (fn, caveat) in METRICS.items():
        pairs = [fn(t, results[t.ticket_id]) for t in tickets]
        low, high, undefined = bootstrap_ci(pairs, resamples, seed)
        out.append(
            MetricResult(
                name=name,
                value=_ratio(pairs),
                ci_low=low,
                ci_high=high,
                numerator=sum(n for n, _ in pairs),
                denominator=sum(d for _, d in pairs),
                undefined_resamples=undefined,
                caveat=caveat,
            )
        )
    return out
