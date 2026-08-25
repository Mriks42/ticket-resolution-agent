# Support Triage Agent

## What this is

An AI agent that resolves customer support tickets end-to-end. It reads a
ticket, decides what information it needs, calls tools in sequence, and either
resolves the ticket or escalates to a human with its reasoning attached.

The product is a subscription SaaS project-management tool with Free/Pro/Team
tiers; tickets cover billing, plan changes, and account issues for that
product.

The point of this project is **measurement**, not feature count. An agent that
resolves 40% of tickets with honest numbers attached is worth more than one that
claims 90% with no eval set. Every capability claim in the README must trace to
a number produced by the eval harness.

## Non-negotiables

- **No mocked results in the eval path.** If a tool can't run yet, the eval
  fails loudly. Never return canned data to make a test pass.
- **No fabricated metrics.** Numbers in the README come from a script in
  `evals/` that I can rerun. If a number can't be reproduced, delete it.
- **Single cloud (AWS).** Do not introduce a second cloud provider.
- **Ask before adding a dependency.** Prefer stdlib and the existing stack.
- **Small diffs.** One concern per change. Stop and report rather than
  continuing into adjacent work I didn't ask for.

## Stack

- Python 3.11, `uv` for dependency management
- FastAPI for the service layer
- Postgres + pgvector for memory and retrieval (docker-compose locally)
- MCP (Model Context Protocol) server exposing agent tools
- LLM calls through a provider-agnostic interface; model name is a config
  value, defaulting to `gpt-4o-mini`
- pytest, with `pytest-asyncio` for the agent loop
- Terraform for AWS provisioning (Lambda or ECS, API Gateway, RDS, S3)
- GitHub Actions for CI
- Structured JSON logs (JSON Lines, append-only) for run tracing — see
  `docs/adr/0001-structured-json-logs-over-mlflow.md`

## Architecture

```
ticket in
  -> agent loop (plan -> select tool -> execute -> observe -> repeat)
       tools exposed over MCP with per-tool permission scopes
       memory layer: prior tickets from this customer + past resolutions
  -> resolve  (draft reply, apply action)
     or escalate (queue for human with reasoning trace)
```

### The agent loop

Explicit state machine, not a while-loop over an LLM call. States: PLAN,
ACT, OBSERVE, RESOLVE, ESCALATE, PENDING_APPROVAL, FAILED. PLAN is
ReAct-style: one step at a time, picking the single next tool call given the
observations so far — not an upfront multi-step plan. Every transition is
logged with a run_id so a full trace can be reconstructed after the fact.

Budgets, enforced in code and configurable:
- max steps per ticket (default 8) — counts tool executions
- max tokens per ticket
- max wall-clock seconds per ticket
Exceeding any budget routes to ESCALATE, never to a silent failure.

### Tools (over MCP)

Read-only, agent calls freely:
- `lookup_customer(customer_id)`
- `search_past_tickets(query, customer_id=None)`
- `retrieve_policy(query)` — returns passages with source IDs

Write / gated, require HITL approval based on deterministic rules — action
type (e.g. any refund over a threshold amount, any reply making a policy
claim) combined with retrieval coverage (whether every claim is backed by a
retrieved passage). Not model self-rated confidence and not logprobs — both
are poorly calibrated and the headline metrics must not rest on them. A
gated action that hasn't been approved resolves to PENDING_APPROVAL, not
RESOLVE or ESCALATE:
- `issue_refund(ticket_id, amount, reason)`
- `send_reply(ticket_id, body)`
- `escalate(ticket_id, reason, trace)`

Every tool call writes an audit record: run_id, tool, args, caller scope,
result status, latency, timestamp. Audit log is append-only.

### Memory

Two layers, kept separate:
- **Episodic**: past tickets and their resolutions for a given customer,
  retrieved by embedding similarity over pgvector.
- **Semantic**: durable facts learned about a customer (plan tier, recurring
  issue, prior refund history), written only on ticket close.

Memory must be switchable off via config. The headline ablation for this
project is resolution rate with memory vs. without, so a stateless baseline has
to stay runnable at all times.

### Guardrail

Any policy claim in a drafted reply must be checkable against the passages
actually retrieved in that run. A claim with no supporting retrieved passage
is a hallucinated-policy event: log it, count it, and route the ticket to
ESCALATE.

## Evaluation

`evals/` is the center of this project, not an afterthought.

Dataset: 20 hand-written, hand-labeled tickets as a permanent held-out sanity
check, expanded synthetically to 150+ for the full eval set. Labels are
drafted by the assistant, then reviewed and corrected before entering the
set — never auto-accepted. Each label carries the correct outcome (resolve,
escalate, or pending_approval), the tools a competent human would have
called, and the ground-truth policy passages relevant to the ticket. The
synthetic generator lives in `evals/generate/` and the README says so
plainly. The policy corpus (15–20 short documents covering refund window,
proration, plan downgrades, failed payments, seat changes, cancellation, and
data retention) lives in `evals/dataset/policies/` with stable IDs.

Metrics, all produced by `evals/run.py`:
- **resolution rate** — resolved without human touch, and correctly so
- **tool-call precision** — fraction of calls that were the right call
- **hallucinated-policy rate** — per the guardrail above
- **escalation precision** — of tickets it escalated, how many genuinely
  needed a human
- **pending-approval rate** — fraction of tickets that hit a gated action and
  required HITL approval; reported on its own, never folded into resolution
  rate or escalation rate
- **cost per resolved ticket** — USD, traced across all steps in the run
- **p50 / p95 latency per resolved ticket**

Report metrics with bootstrap confidence intervals, not bare point estimates.
The eval set is small; a 3-point difference on 180 tickets is probably noise
and the report should make that visible.

Required ablations:
1. memory on vs. off
2. with vs. without the retrieval guardrail
3. step budget sweep (4 / 8 / 16) against resolution rate and cost

## Failure handling

Explicit and tested, one test per case:
- tool returns malformed output
- tool times out
- retrieval returns nothing relevant
- agent repeats the same tool call with the same args (loop detection)
- LLM returns unparseable tool-call JSON
- budget exhausted mid-run

Retries use exponential backoff with a cap. Degradation is graceful: a failed
optional tool narrows the agent's options, it does not crash the run.

## Layout

```
src/
  agent/       loop, state machine, budgets
  tools/       tool implementations
  mcp/         MCP server, permission scopes, audit log
  memory/      episodic + semantic layers
  api/         FastAPI service
evals/
  dataset/     labeled tickets
  generate/    synthetic generator, if used
  run.py       produces every number in the README
  report/      output metrics + CIs
infra/         terraform
tests/
docs/adr/      one file per significant decision
```

## Working agreement

- Write the test before the implementation for anything in `src/agent/` or
  `src/tools/`.
- When a design decision has a real alternative, write a short ADR in
  `docs/adr/` — context, options, choice, what it costs.
- Do not write README performance claims. I write those from eval output.
- If a task seems underspecified, ask rather than guessing. A wrong assumption
  buried three files deep costs more than a question.
- Report at the end of each session: what changed, what's untested, what you
  were unsure about.
