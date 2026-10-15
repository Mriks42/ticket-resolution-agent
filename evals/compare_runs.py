"""Compare configurations, each measured as several runs of `evals/run.py`.

    python -m evals.compare_runs \\
        loop=evals/report/loop_v1.json loop=evals/report/loop_v1_run2.json \\
        gate=evals/report/gate_v1_run1.json gate=evals/report/gate_v1_run2.json

ADR 0005 found that identical runs disagree (temperature 0 and a fixed seed
still flipped 3 of 22 outcomes), so one run is not a measurement of a
configuration. This groups reports by label and shows, per configuration, each
metric for every run and the mean across runs, then which tickets changed
outcome between runs of the *same* configuration.

With `--trace`, it also reads the run trace for the run_ids in each report:
gate decisions by rule, refund amounts by verdict (so "nothing above the
self-serve limit executed" is checkable), escalations by who filed them, and
LLM cost. The trace
is a runtime artifact under var/ and is not committed; the reports are what
make a run comparable later, the trace is what explains it.

The mean across runs is not a confidence interval and is not reported as one.
Each run's own bootstrap CI is in its report; this shows how far apart runs
land, which no single CI captures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTCOME_LETTER = {"resolve": "R", "escalate": "E", "pending_approval": "P"}


def load(pairs: list[str]) -> dict[str, list[dict]]:
    configs: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        label, sep, path = pair.partition("=")
        if not sep:
            raise SystemExit(f"expected label=path, got {pair!r}")
        configs[label].append(json.loads(Path(path).read_text(encoding="utf-8")))
    return configs


def metric_table(configs: dict[str, list[dict]]) -> list[str]:
    names = [m["name"] for m in next(iter(configs.values()))[0]["metrics"]]
    lines = ["", "metric values per run, and the mean across runs (not a CI)", ""]
    for name in names:
        lines.append(f"  {name}")
        for label, reports in configs.items():
            values = [next(m for m in r["metrics"] if m["name"] == name)["value"] for r in reports]
            shown = [v for v in values if v is not None]
            runs = "  ".join("  n/a" if v is None else f"{v:.3f}" for v in values)
            mean = f"{statistics.fmean(shown):.3f}" if shown else "n/a"
            spread = f"{min(shown):.3f}-{max(shown):.3f}" if shown else "n/a"
            lines.append(f"    {label:<10} runs {runs:<28} mean {mean}  range {spread}")
    return lines


def ticket_table(configs: dict[str, list[dict]]) -> list[str]:
    tickets = [p["ticket_id"] for p in next(iter(configs.values()))[0]["per_ticket"]]
    gold = {p["ticket_id"]: p["gold_outcome"] for p in next(iter(configs.values()))[0]["per_ticket"]}
    header = "  ".join(f"{label:<{max(len(label), len(reports))}}" for label, reports in configs.items())
    lines = [
        "",
        "outcome per run (R resolve, E escalate, P pending_approval); * = runs disagree",
        "",
        f"  {'ticket':<7}{'gold':<5}{header}",
    ]
    flips: dict[str, int] = Counter()
    for tid in tickets:
        cells = []
        for label, reports in configs.items():
            letters = "".join(
                OUTCOME_LETTER[next(p for p in r["per_ticket"] if p["ticket_id"] == tid)["agent_outcome"]]
                for r in reports
            )
            unstable = len(set(letters)) > 1
            flips[label] += unstable
            cells.append(f"{letters + ('*' if unstable else ''):<{max(len(label), len(reports))}}")
        lines.append(f"  {tid:<7}{OUTCOME_LETTER[gold[tid]]:<5}" + "  ".join(cells))
    lines += ["", "  tickets whose outcome differs between runs of the same configuration:"]
    lines += [f"    {label:<10} {flips[label]}/{len(tickets)}" for label in configs]
    return lines


def trace_summary(configs: dict[str, list[dict]], trace_path: Path) -> list[str]:
    by_run: dict[str, list[dict]] = defaultdict(list)
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            by_run[event["run_id"]].append(event)

    lines = ["", f"from the trace ({trace_path})", ""]
    for label, reports in configs.items():
        run_ids = [p["run_id"] for r in reports for p in r["per_ticket"]]
        events = [e for rid in run_ids for e in by_run.get(rid, [])]
        missing = sum(1 for rid in run_ids if rid not in by_run)
        gate = Counter(f"{e['tool']}:{e['verdict']}:{e['rule']}" for e in events if e["event"] == "gate")
        filed = Counter(e["filed_by"] for e in events if e["event"] == "escalation_filed")
        cost = [e["cost_usd"] for e in events if e["event"] == "run_finished"]
        lines.append(f"  {label}: {len(reports)} runs, {len(run_ids)} tickets"
                     + (f", {missing} not in this trace" if missing else ""))
        if cost:
            lines.append(f"    LLM cost      ${sum(cost):.4f} total, ${sum(cost) / len(reports):.4f} per run")
        if filed:
            lines.append("    escalations   " + ", ".join(f"filed by {k} {v}" for k, v in sorted(filed.items())))
        for key, n in sorted(gate.items()):
            lines.append(f"    gate          {key:<44} {n}")
        refunds = defaultdict(list)
        for e in events:
            if e["event"] == "gate" and e["tool"] == "issue_refund":
                refunds[e["verdict"]].append(e["args"].get("amount"))
        for verdict, amounts in sorted(refunds.items()):
            numbers = [a for a in amounts if isinstance(a, (int, float))]
            top = f"max ${max(numbers):,.2f}" if numbers else "no amounts"
            lines.append(f"    refunds       {verdict:<10} {len(amounts):>3}  {top}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="label=path to an evals/run.py --out report")
    parser.add_argument("--trace", type=Path, help="var/trace/trace.jsonl, to explain the runs")
    args = parser.parse_args(argv)

    configs = load(args.reports)
    lines = metric_table(configs) + ticket_table(configs)
    if args.trace:
        lines += trace_summary(configs, args.trace)
    print("\n".join(lines + [""]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
