# 1. Structured JSON logs over MLflow for run tracing

## Status

Accepted

## Context

CLAUDE.md left the choice between MLflow and plain structured JSON logs open
for Session 2, to be justified here. Every tool call and state transition in
the agent loop needs to be recorded — run_id, tool, args, caller scope,
result status, latency, timestamp — so a full trace can be reconstructed
after the fact, and so `evals/run.py` can compute resolution rate, tool-call
precision, hallucinated-policy rate, escalation precision, cost, and
latency percentiles with bootstrap CIs.

MLflow is built around experiment tracking: one run per training/evaluation
job, with scalar params and metrics logged against that run, plus artifacts.
That model fits comparing hyperparameter sweeps, not what this project needs.

What this project needs is a per-ticket execution trace — an ordered log of
PLAN/ACT/OBSERVE/RESOLVE/ESCALATE/FAILED transitions and tool calls, all
joined by a single run_id, with enough structure to replay a ticket's full
reasoning path and to feed the eval harness's aggregate metrics.

## Options considered

**MLflow**
- Gives a run-comparison UI and artifact storage out of the box.
- Its data model (one flat set of params/metrics per run) doesn't map well
  onto a nested sequence of state transitions and tool calls per ticket —
  representing a trace would mean either flattening it into metric names/step
  indices (awkward, lossy) or storing the real trace as a JSON artifact
  anyway, at which point MLflow is only wrapping storage.
- Adds a tracking-server dependency (or local file store with its own
  quirks) for something the eval harness would still need to parse back out
  of custom-shaped params.

**Structured JSON logs (JSON Lines to disk)**
- One line per event (state transition or tool call), run_id on every line,
  written append-only as the spec's audit-record requirement already
  demands.
- Directly queryable with `jq` for spot-checking a single ticket's trace, and
  loads straight into pandas for `evals/run.py` and the ablation reports —
  no extra client library, no server process.
- No new infrastructure: it's stdlib `json` + file I/O, consistent with
  "prefer stdlib and the existing stack."

## Decision

Use structured JSON Lines logs, not MLflow. Each event (state transition,
tool call, audit record) is one JSON object per line, tagged with run_id,
written to an append-only log file. `evals/run.py` reads these logs to
produce every metric in the report.

## Consequences

- We give up MLflow's built-in run-comparison UI. For the required
  ablations (memory on/off, guardrail on/off, step-budget sweep), comparison
  views are built directly in `evals/report/` from the JSON logs instead.
- We give up MLflow's artifact versioning/model registry — not needed here,
  since this project isn't training or versioning models.
- Log schema and any migrations are our responsibility; there's no
  framework enforcing structure. This is mitigated by defining the event
  schema once (in `src/agent/` alongside the state machine) and having every
  logging call go through it.
- If a future need arises for comparing many long-running experiments
  side by side (e.g. a large hyperparameter sweep), this decision may be
  revisited — nothing here precludes adding MLflow later for that specific
  purpose without touching the per-ticket trace format.
