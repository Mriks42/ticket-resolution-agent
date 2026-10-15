# 6. The HITL gate: deterministic approval for refunds and replies

## Status

Accepted.

## Context

ADR 0005's loop held every gated call unconditionally, and its first
measurement showed what that hid: on 8 of 10 misses no ESC-001 clause ever
reached the model, which proposed refunds of up to $1,584 without the approval
bands, and resolved a chargeback (T-009) and a non-owner request (T-013).
ADR 0003 had already assigned this to the gate: the bands are arithmetic on a
number the agent holds, not a retrieval target.

CLAUDE.md fixes the shape: approval rests on deterministic rules, action type
combined with retrieval coverage, never model confidence or logprobs, and an
unapproved gated action resolves to PENDING_APPROVAL.

Three questions had real alternatives and were decided with the label owner
before any code was written:

1. **When does a reply need approval?** CLAUDE.md's example, "any reply making
   a policy claim", read literally sends every policy-citing reply to
   PENDING_APPROVAL. That contradicts the 10 gold `resolve` labels, whose
   correct replies cite policy. **Decided: coverage decides.** A reply goes
   out if every clause it cites was retrieved in the run. Citing an unretrieved
   clause is a hallucinated-policy event and escalates, per CLAUDE.md's
   Guardrail section. Catching policy claims made *without* a clause id is the
   guardrail session's job.
2. **The non-owner check (ESC-001.4).** `evals/loader.py` requires every
   non-owner ticket to be labelled escalate. CLAUDE.md's Session 8 notes say a
   non-owner request "is not automatically an escalate". **Decided: block money,
   not answers.** A refund for anyone but the verified owner escalates; a reply
   still goes out. The loader rule stays until Session 8 adds non-owner tickets
   that do not escalate. It is now the stricter of the two, and the known place
   they disagree.
3. **A held refund, then an escalation (T-021).** **Decided: escalation
   wins.** The gate never overrides the model's choice to hand off; the held
   refund goes to the human with the escalation.

## Decisions

**The rules** (`src/agent/gate.py`, pure functions, applied in this order):

| action | rule | clause | verdict |
|---|---|---|---|
| refund | requester is not the verified owner | ESC-001.4 | ESCALATE |
| refund | owner cannot be looked up | ESC-001.4 | ESCALATE (fails closed) |
| refund | reason cites an unretrieved clause | — | ESCALATE |
| refund | ≤ USD 50 | ESC-001.1 | EXECUTE |
| refund | USD 50.01–500 | ESC-001.2 | HOLD for a support manager |
| refund | > USD 500 | ESC-001.3 | HOLD for finance |
| reply | cites an unretrieved clause | — | ESCALATE (hallucinated policy) |
| reply | an approval is pending | — | HOLD with it |
| reply | otherwise | — | EXECUTE → RESOLVE |
| escalate | always | — | EXECUTE → ESCALATE |

The band thresholds are policy copied into code. A test reads ESC-001.1–.3
from the corpus and fails if the text and the constants disagree.

**What each verdict does.**

- **EXECUTE** dispatches the call to a second MCP server granted the write
  scopes. Only the gate reaches it; the model's calls still go to the
  read-only server. Writes are audited like reads, under the same run_id.
- **HOLD** on a refund: the money does not move, and the model is told who
  must approve and shown the clause, so it can tell the customer accurately.
  The run continues. The reply it then writes is held with the refund, so the
  customer is not told before the decision. The ticket ends in
  PENDING_APPROVAL.
- **ESCALATE**: the action does not run; the ticket ends in ESCALATE.

**The owner is verified by the gate, not taken from the model.** The gate does
its own `lookup_customer` through the read server (audited) and compares the
ticket's `requester_email` with the account's `owner_email`, case- and
whitespace-insensitively. Nothing the model wrote enters that comparison.

**The gate shows the model the clause it decided on.** A held refund's result
carries ESC-001.2's or ESC-001.3's text, and an executed one carries ESC-001.1.
Those count as retrieved in the run, so citing them back is covered. This is
the fetch-by-id that ADR 0003 and 0004 asked for, done by the component that
knows which clause applies, through the policy index rather than as a seventh
tool.

**Every escalated or failed run files exactly one escalation.** When the model
escalates, its reason is filed. When the loop or the gate stops a run (budget,
loop detection, invalid calls, the owner check, a hallucinated citation, a
reply that could not be sent, FAILED), the system files it with the stop
reason. Anything held is attached. Before this, a budget stop ended the run in
ESCALATE with nothing queued, and the ticket would have been lost.

**The prompt moves to system_v2.** Only the last paragraph changes: it
described every gated call as held, which is no longer true. It now says
results report what happened and that a reply citing an unretrieved clause
will not be sent. v1 stays on disk; runs record which version they used.

## Measurement

```
python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/gate_v1_runN.json   # x3
python -m evals.compare_runs \
    loop=evals/report/loop_v1.json loop=evals/report/loop_v1_run2.json \
    gate=evals/report/gate_v1_run1.json gate=evals/report/gate_v1_run2.json \
    gate=evals/report/gate_v1_run3.json --trace var/trace/trace.jsonl
```

Same model, seed and retrieval as ADR 0005. Three gated runs against ADR 0005's
two loop-only runs. Per-run values, with the mean across runs (a mean of runs,
not a confidence interval):

| metric | loop only (2 runs) | gated (3 runs) |
|---|---|---|
| resolution rate | 0.318, 0.364 (mean 0.341) | 0.273, 0.364, 0.318 (mean 0.318) |
| escalation precision | 0.600, 0.571 (0.586) | 0.583, 0.700, 0.636 (0.640) |
| tool-call precision | 0.764, 0.761 (0.763) | 0.759, 0.782, 0.769 (0.770) |
| pending-approval rate | 0.000, 0.000 | 0.136, 0.136, 0.136 |
| LLM cost per run | $0.0205 | $0.0175 |

Each gated run's own 95% bootstrap CI on resolution rate is about ±0.18 (run 1:
0.273 [0.091, 0.455]). Runs disagreed on 2 of 22 tickets (T-004, T-008),
against 3 of 22 without the gate.

### What the gate measurably did

These are the gate's own decisions, read from the trace across 66 gated
tickets, not inferences from the headline metrics:

- **No refund above USD 50 executed.** 4 refunds executed, max $50.00, all
  under ESC-001.1. 14 were held (max $1,584.00, for finance). 3 were blocked as
  non-owner. Without the gate, refunds up to $1,584 had been proposed with
  nothing between the proposal and the money.
- **PENDING_APPROVAL became reachable**: 3/22 in every run, T-015, T-017 and
  T-022. Two of those are gold `pending_approval`. T-015 is gold `escalate`, a
  duplicate-billing system fault, and is the "safer wrong outcome" ADR 0005
  predicted: a $132 refund waiting for a manager instead of going out.
- **The non-owner rule fixed T-013** in every run, by its own decision (3 of
  3 `non_owner` blocks). T-017 and T-022 moved from a wrong `resolve` to the
  correct `pending_approval` by band holds.
- **No hallucinated citation was caught.** 0 `citation_not_retrieved`
  decisions in 66 tickets. Every cited clause had been retrieved. That says
  nothing about *uncited* policy claims, which only the guardrail can check.
- **Every escalated run filed exactly one escalation**: 30 by the model, 3 by
  the system (the non-owner blocks).
- **T-021 stays a miss, by decision.** Escalation wins: the model held a $176
  refund and then handed off, and the gate did not override it.

### What it did not do

**The headline metrics did not move beyond run-to-run variation.** Mean
resolution rate fell from 0.341 to 0.318, and mean escalation precision rose
from 0.586 to 0.640. Both differences are smaller than the spread between
identical runs, and far inside each run's ±0.18 interval. The gate converts
*wrong resolves* into holds and escalations; it cannot create correct
resolves, so resolution rate was never going to be where it showed.

**Two changes went in at once, so per-ticket shifts outside the gate's rules
cannot be attributed to it.** Moving the prompt to system_v2 was necessary,
since v1 described behaviour the gate replaced, but it means:

- T-009 (chargeback) moved from `resolve` to a correct `escalate` in all 3
  runs, with **no gate rule firing**. The model escalated on its own, citing the
  chargeback.
- T-004 (gold `resolve`) escalated in 2 of 3 runs. The gate executed its $29
  refund correctly; the model then handed off, in one run claiming the refund
  "requires approval" when the result said it was issued.
- T-019 and T-020 (gold `resolve`, dispositive declines) escalated in every
  run. That is ADR 0005's over-escalation, unchanged.

None of these is a gate decision. Whether system_v2 or chance caused them is
not answerable from five runs.

## Consequences

- PENDING_APPROVAL is now a real outcome, and the pending-approval rate means
  something. Its base is still n=3, so it moves in 33-point steps; that is the
  Session 8 target, not something the gate can fix.
- Escalations reach the store, so "queue for human" is no longer only a state
  name.
- The loader's non-owner rule is now stricter than the gate. Loosen it when
  Session 8 adds non-owner tickets that should not escalate.
- `AgentLoop` requires a `writer`; anything constructing one must supply the
  write-scoped executor.

## Open

- **Refunds are banded one at a time.** In one run the model refunded exactly
  $50 on T-021, a $176 charge, and the gate executed it, because the gate judges
  each refund in isolation. Nothing stops $50 + $49 on one ticket. Banding the
  *cumulative* refund per ticket would close this. That is a reading of ESC-001.1
  ("a refund of up to USD 50"), so it is the label owner's call, and it is the
  first thing to decide next.
- **Over-escalation of dispositive declines** (T-008, T-019, T-020) and the
  executed-refund misreading on T-004 are prompt and behaviour work, for the
  Session 8 synthetic set, not these 22.
- **The guardrail**: detecting policy claims made without a clause id, and the
  hallucinated-policy rate as a reported metric. The gate's coverage check only
  sees citations.
