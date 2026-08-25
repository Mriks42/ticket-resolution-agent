"""Run an agent against the labeled ticket set and report metrics with CIs.

    python -m evals.run
    python -m evals.run --agent evals.agents.stub:AlwaysEscalateAgent --out report.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evals.loader import POLICY_DIR, TICKETS_PATH, load_tickets
from evals.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, MetricResult, evaluate
from evals.schema import Agent, AgentResult, Confidence, Outcome, Ticket


def load_agent(spec: str) -> Agent:
    """Instantiate an agent from a 'package.module:ClassName' spec."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"agent spec must be 'module:ClassName', got {spec!r}")
    agent = getattr(importlib.import_module(module_name), attr)()
    if not isinstance(agent, Agent):
        raise TypeError(f"{spec} does not implement the Agent protocol")
    return agent


async def run_agent(agent: Agent, tickets: list[Ticket]) -> dict[str, AgentResult]:
    results: dict[str, AgentResult] = {}
    for ticket in tickets:
        result = await agent.run(ticket)
        if result.ticket_id != ticket.ticket_id:
            raise ValueError(
                f"{agent.name} returned ticket_id {result.ticket_id!r} "
                f"for ticket {ticket.ticket_id!r}"
            )
        results[ticket.ticket_id] = result
    return results


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def format_report(
    agent_name: str,
    tickets: list[Ticket],
    results: dict[str, AgentResult],
    metrics: list[MetricResult],
    resamples: int,
    seed: int,
) -> str:
    gold = Counter(t.labels.outcome.value for t in tickets)
    pred = Counter(results[t.ticket_id].outcome.value for t in tickets)
    unreviewed = [t for t in tickets if t.labels.confidence is not Confidence.HIGH]

    lines = [
        "",
        f"agent      {agent_name}",
        f"tickets    {len(tickets)}",
        f"bootstrap  {resamples} resamples, seed {seed}, percentile 95% CI",
        "",
        "gold outcomes      " + "  ".join(
            f"{o.value}={gold.get(o.value, 0)}" for o in Outcome
        ),
        "agent outcomes     " + "  ".join(
            f"{o.value}={pred.get(o.value, 0)}" for o in Outcome
        ),
        "",
        f"{'metric':<24}{'value':>8}  {'95% CI':<18}{'n':>12}",
        "-" * 64,
    ]
    for m in metrics:
        ci = (
            "n/a"
            if m.ci_low is None
            else f"[{m.ci_low:.3f}, {m.ci_high:.3f}]"
        )
        n = f"{m.numerator:g}/{m.denominator:g}"
        lines.append(f"{m.name:<24}{_fmt(m.value):>8}  {ci:<18}{n:>12}")

    lines += ["", "how to read these"]
    for m in metrics:
        lines.append(f"  {m.name}:")
        for chunk in _wrap(m.caveat, 68):
            lines.append(f"    {chunk}")

    warnings: list[str] = []
    for m in metrics:
        if m.denominator == 0:
            warnings.append(
                f"{m.name} is undefined (denominator 0), not 0.0. Nothing to measure."
            )
        elif m.is_degenerate:
            warnings.append(
                f"{m.name} has a zero-width CI: every resample gave the same value. "
                "That means the metric is constant across the set, not that it is "
                "precisely estimated."
            )
        if m.undefined_resamples:
            warnings.append(
                f"{m.name}: {m.undefined_resamples}/{resamples} resamples had an "
                "empty denominator and were dropped from the CI."
            )
    if unreviewed:
        ids = ", ".join(t.ticket_id for t in unreviewed)
        warnings.append(
            f"{len(unreviewed)} of {len(tickets)} labels sit below high "
            f"confidence ({ids}); each records why in its label_note. Every "
            "metric above inherits that uncertainty and no CI accounts for it."
        )
    if len(tickets) < 150:
        warnings.append(
            f"{len(tickets)} tickets is far below the 150-200 the project calls "
            "for. CIs this wide cannot separate agents that differ by less than "
            "tens of points."
        )

    if warnings:
        lines += ["", "warnings"]
        for w in warnings:
            wrapped = _wrap(w, 68)
            lines.append(f"  ! {wrapped[0]}")
            lines += [f"    {c}" for c in wrapped[1:]]

    lines.append("")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def build_report(
    agent_name: str,
    tickets: list[Ticket],
    results: dict[str, AgentResult],
    metrics: list[MetricResult],
    resamples: int,
    seed: int,
) -> dict:
    return {
        "agent": agent_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket_count": len(tickets),
        "bootstrap": {"resamples": resamples, "seed": seed, "ci": 0.95},
        "gold_outcomes": dict(Counter(t.labels.outcome.value for t in tickets)),
        "agent_outcomes": dict(
            Counter(results[t.ticket_id].outcome.value for t in tickets)
        ),
        "unreviewed_labels": [
            t.ticket_id for t in tickets if t.labels.confidence is not Confidence.HIGH
        ],
        "metrics": [m.as_dict() for m in metrics],
        "per_ticket": [
            {
                "ticket_id": t.ticket_id,
                "gold_outcome": t.labels.outcome.value,
                "agent_outcome": results[t.ticket_id].outcome.value,
                "label_confidence": t.labels.confidence.value,
                "expected_tools": [x.value for x in t.labels.expected_tools],
                "called_tools": [
                    c.tool.value for c in results[t.ticket_id].tool_calls
                ],
            }
            for t in tickets
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="evals.agents.stub:AlwaysEscalateAgent")
    parser.add_argument("--dataset", type=Path, default=TICKETS_PATH)
    parser.add_argument("--policies", type=Path, default=POLICY_DIR)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)

    tickets = load_tickets(args.dataset, args.policies)
    agent = load_agent(args.agent)
    results = asyncio.run(run_agent(agent, tickets))
    metrics = evaluate(tickets, results, args.resamples, args.seed)

    print(
        format_report(
            agent.name, tickets, results, metrics, args.resamples, args.seed
        )
    )

    if args.out:
        report = build_report(
            agent.name, tickets, results, metrics, args.resamples, args.seed
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
