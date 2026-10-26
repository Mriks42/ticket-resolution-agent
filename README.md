# Ticket Resolution Agent

An AI agent that resolves customer support tickets end-to-end. It reads a
ticket, decides what it needs to know, calls tools one step at a time, and
either resolves the ticket, holds an action for human approval, or escalates
to a human with its reasoning attached.

The tickets are billing, plan-change and account requests for a subscription
project-management product with Free, Pro and Team plans.

**The point of this project is measurement, not feature count.** Every
capability claim here traces to a number produced by a script in `evals/` that
can be rerun. Where a number has not been produced, this README says so rather
than estimating it.

## Results

> Written by the maintainer from eval output. Every figure in this section
> must come from one of the commands below; if it cannot be reproduced, it is
> deleted.

<!-- Results go here. Commands that produce them:
       python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/<name>.json
       python -m evals.compare_runs <label>=<report> ... --trace var/trace/trace.jsonl
       python -m evals.retrieval_baseline --backend lexical dense hybrid --floor 0.30
       python -m evals.guardrail_eval
-->

Before reading any figure, three things about the eval set:

- **It is small.** 22 hand-written tickets. A 95% confidence interval on a
  rate spans roughly ±20 points at this size, so most differences between two
  configurations are unproven rather than small wins. Every metric is reported
  with a bootstrap interval for this reason.
- **Runs are not repeatable.** Temperature 0 and a fixed seed still change
  some outcomes between identical runs ([ADR 0005](docs/adr/0005-agent-loop.md)),
  so a configuration is measured as several runs, compared with
  `evals.compare_runs`.
- **Two label sets are awaiting review.** The retrieval probes and the
  guardrail's validation replies were drafted by an assistant and are marked
  `"reviewed": false`. Figures that depend on them are provisional until a
  human signs off.

## How it works

```
ticket ─► agent loop ── PLAN ─► ACT ─► OBSERVE ─┐   one tool call per step
              ▲                                  │
              └──────────────────────────────────┘
                     │ reads                │ refunds, replies, escalations
                     ▼                      ▼
             MCP server (read scopes)   guardrail ─► gate ─► MCP server (write scopes)
             lookup_customer            every policy  bands,        only the gate
             search_past_tickets        figure must   owner check,  reaches it
             retrieve_policy ◄─ pgvector be grounded  approval
                     │
                     ▼
        RESOLVE · PENDING_APPROVAL · ESCALATE · FAILED   (every hop logged with a run_id)
```

**The agent loop** ([ADR 0005](docs/adr/0005-agent-loop.md)) is an explicit
state machine, not a while-loop around an LLM call. The model is offered the
six tools as native function calls and must pick exactly one per turn. Step,
token and wall-clock budgets, loop detection and repeated invalid calls all end
a run in ESCALATE with a stated reason, never in a silent failure. Every
transition, model call, observation and gate decision is written to an
append-only JSON Lines trace, including what the model was shown, so a run can
be reconstructed after the fact ([ADR 0001](docs/adr/0001-structured-json-logs-over-mlflow.md)).

**Tools** are served over MCP by a stdlib JSON-RPC server with per-tool
permission scopes and an audit record for every call, refused ones included
([ADR 0002](docs/adr/0002-stdlib-mcp-server-over-official-sdk.md)). The
model's calls reach a server granted read scopes only. Writes go to a separate
server that only the gate dispatches to.

**Retrieval** ([ADR 0003](docs/adr/0003-retrieval-approach.md),
[ADR 0004](docs/adr/0004-dense-retrieval-and-similarity-floor.md)) returns
clause-level passages (`REF-001.3`, not `POL-REF-001`), so every policy claim
can be checked against the specific clause behind it. The configured backend is
dense retrieval over pgvector with a calibrated similarity floor, so "no policy
covers this" can still come back as an empty result. A lexical TF-IDF backend
stays as the baseline every change is measured against, and a hybrid backend is
kept for ablations.

**The retrieval guardrail** ([ADR 0007](docs/adr/0007-retrieval-guardrail.md))
checks every reply and refund reason before anything happens. Each figure
(amounts, durations, seat counts, percentages) and each clause citation must be
grounded in a passage retrieved in that run or in the customer's own data. A
sentence citing a clause is held to that clause. The check is deterministic: no
model judges another model's claims. A violation is a hallucinated-policy event,
which blocks the action and escalates by default. Its known blind spot is a
wrong claim with no figure in it; the validation set measures how much that
misses.

**The HITL gate** ([ADR 0006](docs/adr/0006-hitl-gate.md)) decides gated
actions by deterministic rules, never by model confidence:

| action | rule | result |
|---|---|---|
| refund | requester is not the verified account owner (ESC-001.4) | escalate |
| refund | ticket's refund total ≤ USD 50 (ESC-001.1) | executes |
| refund | ticket's refund total USD 50.01–500 (ESC-001.2) | held for a support manager |
| refund | ticket's refund total > USD 500 (ESC-001.3) | held for finance |
| reply | an approval is pending on the ticket | held with it |
| reply | otherwise | sent |
| escalate | always | filed; wins over anything held |

The band thresholds are policy copied into code; a test fails if they drift
from the policy text. The gate verifies the owner with its own audited lookup,
never from anything the model said. Writes are never retried, and a refund
that times out escalates as an unknown outcome. Every escalated or failed run
files exactly one escalation, so no stopped ticket is lost.

## Quickstart

Requirements: Python 3.11, [uv](https://docs.astral.sh/uv/), and Docker for
Postgres + pgvector.

```sh
uv sync                       # dependencies, including the test group
cp .env.example .env          # then set OPENAI_API_KEY
docker compose up -d          # Postgres 17 + pgvector, pinned
uv run pytest                 # the full suite; no network, no key, $0
```

Tests that need Postgres skip with the reason shown when it is not running.
One test calls the real model and is opt-in:

```sh
TRA_LIVE=1 uv run pytest tests/test_live.py -s    # one ticket, about $0.001
```

## Running the evals

| command | what it measures | cost |
|---|---|---|
| `uv run python -m evals.run` | the always-escalate stub: the floor any agent must beat | $0 |
| `uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/run1.json` | the real agent over all tickets, five metrics with bootstrap CIs | a few cents |
| `uv run python -m evals.compare_runs a=<report> b=<report> --trace var/trace/trace.jsonl` | configurations compared as sets of runs, plus gate and guardrail decisions from the trace | $0 |
| `uv run python -m evals.retrieval_baseline --backend lexical dense hybrid --floor 0.30` | retrieval recall per backend, no agent involved | first run only, under a cent |
| `uv run python -m evals.retrieval_baseline --backend lexical dense --calibrate-floor` | the dense similarity floor, by a rule fixed in advance | $0 after the first run |
| `uv run python -m evals.guardrail_eval` | the guardrail's precision and recall on labelled reply drafts | $0 |

Agent runs are capped by `EVAL_SPEND_CAP_USD` (default $1.00). A run that would
exceed it aborts rather than reporting partial numbers. Embeddings are cached in
Postgres by model and text hash, so retrieval reruns make no API calls and rank
identically. Reports go to `evals/report/` and run traces and audit logs to
`var/`; both are gitignored.

**Ablations** run as the same command under different settings, e.g. the
guardrail with and without blocking:

```sh
GUARDRAIL_MODE=block  uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/block1.json
GUARDRAIL_MODE=detect uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/detect1.json
```

## Evaluation

**Dataset** (`evals/dataset/`):

- `tickets.json`: 22 hand-written tickets (10 resolve, 9 escalate, 3
  pending_approval). Each is labelled with the correct outcome, the tools a
  competent human would call, and the relevant policy documents. They range from
  clean questions to scrappy prose with the key detail buried mid-message. They
  are the permanent held-out check, and the prompt is not tuned on them.
- `policies/`: 20 policy documents, 104 citable clauses, with stable IDs.
  `POLICY_NOTES.md` documents one deliberate contradiction and several
  deliberate gaps, which test whether the agent notices rather than improvises.
  The notes are kept out of the retrieval corpus.
- `customers.json`: account records and history behind `lookup_customer` and
  `search_past_tickets`.
- `retrieval_probes.json`: 16 queries no clause answers, for the similarity
  floor. **Unreviewed.**
- `guardrail_cases.json`: 42 reply drafts with planted errors, for the
  guardrail. **Unreviewed.**

**No synthetic tickets exist yet.** CLAUDE.md plans to expand the set to 150+
with a generator in `evals/generate/`, which is currently empty. The targets
are fixed in advance: a larger pending-approval base, a spread of non-owner
requests, and the same messy writing as the hand-written set.

**Metrics** reported by `evals/run.py`, each with a 95% bootstrap interval over
tickets:

| metric | definition |
|---|---|
| resolution rate | resolved, correctly, without human touch, over all tickets |
| escalation precision | of tickets escalated, how many genuinely needed a human |
| tool-call precision | fraction of tool calls that were a right call (name-level) |
| pending-approval rate | tickets that ended waiting for approval; never folded into the others |
| hallucinated-policy rate | drafted replies with a guardrail violation, over drafted replies |

Not yet reported: **cost per resolved ticket** (the trace records LLM cost per
run, but embedding cost is not yet included) and **p50/p95 latency per resolved
ticket** (recorded in the trace, not yet a metric).

## Configuration

Set in `.env` (see `.env.example`); the environment overrides the file.

| variable | default | meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | the model and the embeddings |
| `DATABASE_URL` | compose service | Postgres + pgvector |
| `RETRIEVAL_BACKEND` | `lexical` in code, `dense` in `.env.example` | `lexical`, `dense` or `hybrid` |
| `RETRIEVAL_DENSE_FLOOR` | — | cosine floor; required for `dense` and `hybrid` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | recorded with every report |
| `LLM_MODEL` | `gpt-4o-mini` | recorded with every run, with the served snapshot |
| `GUARDRAIL_MODE` | `block` | `block` or `detect` (logs and counts, does not block) |
| `EVAL_SPEND_CAP_USD` | `1.00` | hard cap per eval run |
| `AGENT_MAX_STEPS` | `8` | tool decisions per ticket |
| `AGENT_MAX_TOKENS` | `50000` | input + output tokens per ticket |
| `AGENT_MAX_WALL_CLOCK_SECONDS` | `120` | per ticket |
| `TRA_VAR_DIR` | `var/` | where traces, audit logs and the write store go |
| `MEMORY_ENABLED` | `true` | `false` keeps all customer history from the agent: the stateless baseline (ADR 0008) |
| `LOG_LEVEL` | — | reserved; not read by any code yet |

## Layout

```
src/
  agent/      loop, state machine, budgets, gate, guardrail, LLM adapter, prompts
  tools/      tool implementations, retrieval backends, embeddings
  mcp/        MCP server, permission scopes, audit log
  memory/     (not built yet)
  api/        (not built yet)
evals/
  dataset/    tickets, policy corpus, customers, probes, guardrail cases
  generate/   (empty: no synthetic generator yet)
  run.py      every agent metric
  retrieval_baseline.py, guardrail_eval.py, compare_runs.py
infra/        (not built yet: Terraform for AWS)
docs/adr/     one file per significant decision
tests/
```

## Status

Built and measured: the retrieval backends, the agent loop, the HITL gate, the
retrieval guardrail, and the eval harness with bootstrap CIs. Each ADR records
what was measured, including the results that did not go the expected way.

Not built yet:

- the memory layer (episodic and semantic) and its on/off ablation
- the cost-per-resolved-ticket and latency metrics
- the step-budget sweep (4 / 8 / 16)
- the synthetic dataset expansion
- the FastAPI service, Terraform for AWS, and CI

## Design decisions

| ADR | decision |
|---|---|
| [0001](docs/adr/0001-structured-json-logs-over-mlflow.md) | structured JSON Lines logs over MLflow for run tracing |
| [0002](docs/adr/0002-stdlib-mcp-server-over-official-sdk.md) | a stdlib JSON-RPC MCP server over the official SDK |
| [0003](docs/adr/0003-retrieval-approach.md) | embedding retrieval over pgvector, built before the agent loop |
| [0004](docs/adr/0004-dense-retrieval-and-similarity-floor.md) | dense retrieval as the configured backend, with a calibrated floor |
| [0005](docs/adr/0005-agent-loop.md) | the agent loop: native tool calls, budgets, failure handling |
| [0006](docs/adr/0006-hitl-gate.md) | the HITL gate: deterministic approval for refunds and replies |
| [0007](docs/adr/0007-retrieval-guardrail.md) | the retrieval guardrail: a deterministic check of every policy figure |
