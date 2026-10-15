# 5. The agent loop: native tool calls, held gated actions, first measurement

## Status

Accepted. The numbers below are the loop *without* the HITL gate or the
guardrail, which come next, and they read accordingly. Holding every gated call
unconditionally is superseded by the gate in ADR 0006; everything else here
stands.

## Context

CLAUDE.md specifies the loop: an explicit state machine (PLAN, ACT, OBSERVE,
RESOLVE, ESCALATE, PENDING_APPROVAL, FAILED), one tool call per step, three
budgets that route to ESCALATE, every transition logged with a run_id, and six
failure cases tested one by one. It leaves the mechanics open. These were
decided with the label owner before any code was written:

- **Native function calling**, not JSON-in-prose. The six catalog schemas go to
  the provider as tools, so the catalog stays the single source of truth for
  what a tool accepts.
- **Scripted LLM in tests, plus one opt-in live test.** Unit tests are $0 and
  deterministic. `tests/test_live.py` runs one real ticket when `TRA_LIVE=1`.
  Eval runs always use the real model.
- **Loop only. Gated calls are held, never executed.** The HITL gate is the
  next session.

## Decisions

**One step at a time, enforced by the provider.** `tool_choice="required"`
with `parallel_tool_calls=false`. If a model sends extra calls anyway, only
the first runs, and the others are answered `not_executed` so the
conversation stays well-formed.

**Gated calls are held as proposals.** `issue_refund` is held and the run
continues. `send_reply` ends the run in RESOLVE, `escalate` in ESCALATE. The
model is told a held refund "has not been issued" so it does not claim
otherwise. PENDING_APPROVAL is in the state table but unreachable until the gate.

**Reads go through the real MCP server, in-process, with read scopes only.**
The loop sends JSON-RPC `tools/call` to `MCPServer.handle` with the run_id in
`_meta`, so every read is permission-checked and audited exactly as over
stdio. Write scopes are never granted. A gated call that reached the server by
mistake would be refused, and the refusal audited (tested). In-process rather
than over a stdio subprocess keeps the dense index and its connection in one
process; stdio is still exercised by the MCP tests.

**What ends a run, and how.**

| condition | result |
|---|---|
| model proposes `send_reply` / `escalate` | RESOLVE / ESCALATE |
| step, token or wall-clock budget exhausted | ESCALATE |
| identical call (same tool, same args) | first repeat not re-run, the model is told it has the result; 2nd repeat → ESCALATE |
| unparseable args, unknown tool, or no tool call | first is reported back for correction; 2nd → ESCALATE |
| tool error, malformed output, timeout | an observation; the run continues with fewer options |
| no usable completion from the provider | FAILED, reported as an escalation with `final_state="FAILED"` so it stays countable |
| spend cap exceeded | the eval run aborts. Not an outcome, deliberately |

**Budgets.** Steps default to 8 (CLAUDE.md). Tokens (50,000) and wall-clock
(120 s) are starting points, not measured choices; the step-budget sweep
should revisit all three. Steps count tool decisions, so a held refund counts
as one.

**The prompt is versioned and holds no policy.** `src/agent/prompts/system_v1.md`
contains behaviour rules only. Policy reaches the model through
`retrieve_policy` and nothing else, which is what lets the guardrail check every
claim against a retrieved passage. Every run records the prompt version and the
SHA-256 of its text.

**system_v1 was written before any eval run and has not been revised since.**
The 22 hand-written tickets are the permanent held-out check (CLAUDE.md).
Tuning the prompt on them would turn that check into a training set. Prompt
iteration belongs on the Session 8 synthetic set.

**The trace records what the model saw, verbatim.** Each observation event
carries the exact content returned to the model, written at the one point in
the code where that content is produced, so the two cannot drift. Run 1 below
predates this; its outcomes are unaffected (the change was logging only), but
its trace cannot show observation content.

**Spend.** $1.00 hard cap per eval run (`EVAL_SPEND_CAP_USD`). Priced from
OpenAI's model page, checked 2026-10-14: gpt-4o-mini $0.15 / $0.60 per 1M
input / output tokens. All input is priced at the full rate, ignoring the
cached-input discount, which overstates cost slightly.

## First measurement

```
python -m evals.run --agent src.agent.loop:LoopAgent
```

gpt-4o-mini (served `gpt-4o-mini-2024-07-18`), temperature 0, seed 20260824,
dense retrieval at floor 0.30, 22 tickets. Two identical runs:

| metric | stub (always escalate) | loop, run 1 | loop, run 2 |
|---|---|---|---|
| resolution rate | 0.000 | 0.318 [0.136, 0.500] (7/22) | 0.364 [0.182, 0.545] (8/22) |
| escalation precision | 0.409 [0.227, 0.591] | 0.600 [0.286, 0.900] (6/10) | 0.571 [0.167, 1.000] (4/7) |
| tool-call precision | 0.409 | 0.764 [0.682, 0.837] | 0.761 [0.667, 0.850] |
| pending-approval rate | 0.000 | 0.000 | 0.000 |
| LLM cost, 22 tickets | — | $0.0206 | $0.0204 |

Median 5.3 s per ticket (run 1). No run hit a budget, tripped loop detection,
made an invalid call or ended in FAILED. Every run ended because the model
proposed a reply or an escalation, in at most 6 of 8 steps.

### How to read it

- **Resolution rate is capped at 10/22 (0.455)**, the gold resolve share. The
  pending-approval rate is 0 because the gate does not exist, so the three
  `pending_approval` tickets cannot be scored right by construction.
- **Temperature 0 and a fixed seed do not make runs repeatable.** 3 of 22
  outcomes flipped between identical runs (T-015, T-016, T-019). One ticket is
  4.5 points of resolution rate, so a single run's number carries
  nondeterminism *on top of* the ±20-point sampling interval CLAUDE.md
  describes. Agent-to-agent comparisons should average several runs per
  configuration. Single runs are not evidence.
- **The cost is LLM tokens only.** Embedding the queries the agent writes is
  not charged to the spend guard or counted here. At $0.02 per 1M tokens it is
  negligible, but the cost metric must include it before it is reported.

### Where it fails (run 2, the run with a full trace)

Of 10 misses, **8 never saw any ESC-001 clause.** Across the misses the model
proposed refunds of $88, $88, $132, $264, $290 and **$1,584**, and on none of
those runs did the approval bands (ESC-001.1–.3) reach it. It resolved T-009
(chargeback already filed) and T-013 (request from a non-owner) without
ESC-001.4's mandatory-escalation list ever coming back from a query.

This is ADR 0003's class (B), arriving on schedule. ESC-001 is reached by
applying a rule to a value the agent already holds, not by resembling the
ticket. ADR 0003 assigned it to the gate, and that holds. These are
**predictions for the gate session, not measurements**:

- Band arithmetic on the proposed amount (every refund above is over $50)
  routes every one of those refunds to PENDING_APPROVAL. That is the right
  outcome for T-017 and T-022, and a *safer* wrong outcome for T-009, T-013,
  T-015 and T-016: a human approves instead of money moving.
- ESC-001.4's non-owner trigger is mechanical. POLICY_NOTES says so, and
  `owner_email` / `requester_email` exist to make it so. It belongs in the gate,
  not in the hope that a query surfaces it.

The other misses run the opposite way: **over-escalation of dispositive
declines.** T-008 and T-020 (gold resolve) escalated in both runs; T-019
flipped. The model treats "policy says no" as "a human should decide". That is
a prompt and behaviour question, to be worked on the synthetic set, not on these
22.

## Consequences

- `AgentResult` gains `final_state` and `stop_reason` (optional, so the stub
  agent is unchanged). `evals/run.py` prints final states and records them per
  ticket.
- `evals/run.py` loads `.env`; the real agent needs the API key and, on the
  dense backend, Postgres.
- The retry policy is shared between the embedder and the chat client
  (`src/httpjson.py`).
- A timed-out tool call's worker thread is abandoned, not killed. Harmless for
  reads; it will need revisiting if a tool ever writes.

## Open

- **Gate session:** band arithmetic and the owner check (above), and what a
  held refund followed by an escalation should become (T-021).
- **Cost metric:** add embedding cost; report cost per *resolved* ticket.
- **Latency:** p50/p95 per resolved ticket is in the trace (`elapsed_s`) but
  not yet a metric in `evals/run.py`.
- **Variance:** repeat runs per configuration before any comparison.
