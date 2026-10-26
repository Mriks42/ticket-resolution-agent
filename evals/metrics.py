"""Metric definitions and percentile bootstrap CIs.

Most metrics are a ratio of sums. Each ticket contributes a (numerator,
denominator) pair, and the metric is sum(num) / sum(den). The latency metrics
are percentiles instead: each ticket contributes a value or nothing, and the
metric is a percentile of the values. Either way the bootstrap resamples whole
tickets, which is the right unit here: a ticket is the independent
observation, not a tool call.

Cost and latency need the agent to report them (`AgentResult.cost_usd`,
`elapsed_s`). An agent that does not gets `n/a`, never a number (ADR 0009).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from evals.schema import AgentResult, Outcome, Ticket, Tool

Contribution = tuple[float, float]
MetricFn = Callable[[Ticket, AgentResult], Contribution]
#: A ticket's value for a percentile metric, or None if it is not in the sample.
SampleFn = Callable[[Ticket, AgentResult], float | None]

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


def _correctly_resolved(ticket: Ticket, result: AgentResult) -> bool:
    return result.outcome is Outcome.RESOLVE and ticket.labels.outcome is Outcome.RESOLVE


def _cost_per_resolved(ticket: Ticket, result: AgentResult) -> Contribution:
    """Every ticket's spend over correct resolutions (ADR 0009)."""
    return (result.cost_usd, 1.0 if _correctly_resolved(ticket, result) else 0.0)


def _resolved_latency(ticket: Ticket, result: AgentResult) -> float | None:
    return result.elapsed_s if _correctly_resolved(ticket, result) else None


def _hallucinated_policy_rate(ticket: Ticket, result: AgentResult) -> Contribution:
    drafted = any(call.tool is Tool.SEND_REPLY for call in result.tool_calls)
    if not drafted:
        return (0.0, 0.0)
    return (1.0 if result.policy_violations else 0.0, 1.0)


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
    "hallucinated_policy_rate": (
        _hallucinated_policy_rate,
        "Drafted replies with at least one policy fact the deterministic "
        "guardrail could not ground (ADR 0007), over drafted replies. A "
        "flagged reply counts whether it was blocked or sent. It sees "
        "amounts, durations, counts, percentages and clause citations only: "
        "a wrong claim with no number in it is invisible here, so read this "
        "beside the guardrail's recall from evals.guardrail_eval.",
    ),
    "cost_per_resolved_usd": (
        _cost_per_resolved,
        "Total spend on every ticket, divided by tickets correctly resolved, "
        "so spend on escalations and wrong resolutions is charged to the "
        "resolutions. LLM tokens at list price (cached input not discounted, "
        "so slightly high) plus embedding calls actually made; text already "
        "in the embedding cache costs nothing, so a rerun can cost a little "
        "less than a cold run. Undefined when nothing is resolved.",
    ),
}

_LATENCY_CAVEAT = (
    "Wall clock per correctly resolved ticket, start to terminal state: model, "
    "tool and embedding calls together. With about ten resolved tickets, p95 "
    "is close to the slowest of them and its interval is wide. Provider load "
    "moves it from day to day, so compare latency within a session."
)

#: name -> (value per ticket, percentile, caveat).
PERCENTILE_METRICS: dict[str, tuple[SampleFn, float, str]] = {
    "latency_p50_resolved_s": (_resolved_latency, 0.50, _LATENCY_CAVEAT),
    "latency_p95_resolved_s": (_resolved_latency, 0.95, _LATENCY_CAVEAT),
}

#: The AgentResult field a metric cannot be computed without.
_REQUIRES = {
    "cost_per_resolved_usd": "cost_usd",
    "latency_p50_resolved_s": "elapsed_s",
    "latency_p95_resolved_s": "elapsed_s",
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


def _sample_percentile(values: list[float | None], q: float) -> float | None:
    present = sorted(v for v in values if v is not None)
    return _percentile(present, q) if present else None


def bootstrap_ci(
    pairs: list[Contribution],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, int]:
    """Percentile bootstrap of a ratio of sums over tickets. Returns
    (low, high, undefined_count)."""
    return _bootstrap(pairs, _ratio, resamples, seed, alpha)


def _bootstrap(
    items: list,
    statistic: Callable[[list], float | None],
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, int]:
    if not items:
        return (None, None, resamples)

    rng = random.Random(seed)
    n = len(items)
    values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
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
        if not _reported(name, tickets, results):
            out.append(_not_reported(name, caveat))
            continue
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
    for name, (fn, q, caveat) in PERCENTILE_METRICS.items():
        if not _reported(name, tickets, results):
            out.append(_not_reported(name, caveat))
            continue
        values = [fn(t, results[t.ticket_id]) for t in tickets]
        statistic = lambda sample, q=q: _sample_percentile(sample, q)  # noqa: E731
        low, high, undefined = _bootstrap(values, statistic, resamples, seed)
        out.append(
            MetricResult(
                name=name,
                value=statistic(values),
                ci_low=low,
                ci_high=high,
                # Tickets in the sample, out of all tickets.
                numerator=float(sum(v is not None for v in values)),
                denominator=float(len(values)),
                undefined_resamples=undefined,
                caveat=caveat,
            )
        )
    return out


def _reported(name: str, tickets: list[Ticket], results: dict[str, AgentResult]) -> bool:
    field = _REQUIRES.get(name)
    return field is None or all(getattr(results[t.ticket_id], field) is not None for t in tickets)


def _not_reported(name: str, caveat: str) -> MetricResult:
    return MetricResult(
        name=name, value=None, ci_low=None, ci_high=None, numerator=0.0, denominator=0.0,
        undefined_resamples=0,
        caveat=f"Not reported by this agent: its results carry no {_REQUIRES[name]}. {caveat}",
    )


def format_value(name: str, value: float | None) -> str:
    """A metric value in its own units: rates to 3 places, USD to 5 (a
    ticket costs a fraction of a cent), seconds to 2."""
    if value is None:
        return "n/a"
    if name.endswith("_usd"):
        return f"{value:.5f}"
    if name.endswith("_s"):
        return f"{value:.2f}"
    return f"{value:.3f}"
