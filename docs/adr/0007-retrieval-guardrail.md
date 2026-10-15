# 7. The retrieval guardrail: a deterministic check of every policy figure

## Status

Accepted. The reply set that validates the guardrail is drafted and awaiting
the label owner's review; every guardrail precision and recall figure below is
provisional until then.

## Context

CLAUDE.md: *"Any policy claim in a drafted reply must be checkable against the
passages actually retrieved in that run. A claim with no supporting retrieved
passage is a hallucinated-policy event: log it, count it, and route the ticket
to ESCALATE."* It also requires a hallucinated-policy rate among the headline
metrics, and a "with vs without the retrieval guardrail" ablation.

ADR 0006's gate already escalated a reply citing a clause that was never
retrieved. Two failures got past it: a policy claim made *without* a citation,
and a claim that cites a retrieved clause but misstates it ("within 30 days
(REF-001.3)", where REF-001.3 says 14).

Three choices were made with the label owner before any code was written:

1. **Detection is deterministic.** Checkable facts are extracted from the reply
   and each must be grounded in what the run actually saw. No model judges
   another model's claims. CLAUDE.md says the headline metrics must not rest on
   model judgement, and a second model would also add cost to every reply. An
   LLM judge stays possible later, as a separately measured layer.
2. **The guardrail is validated against labelled reply drafts.** These are
   cases with planted errors, drafted by the assistant and reviewed by the label
   owner, so the hallucinated-policy rate comes with a precision and recall
   beside it.
3. **"Without guardrail" means detect but don't block.** The check still runs
   and counts, but the reply goes out, so both arms of the ablation report the
   same rate.

## Decisions

**What is checked** (`src/agent/guardrail.py`). The reply is split into
sentences. From each sentence it takes the clause ids it cites and every
normalised figure: money (USD 50, $1,584), durations (14 days, 30-day, two
weeks = 14 days, 5–10 business days, 4 business hours), counts (3 seats,
3-seat), storage (500 MB) and percentages. Refund reasons are checked the same
way. Escalation reasons are not: they are addressed to a human, not the
customer.

**What grounds a figure.** It must appear, with the same value and unit, in
either:

- a passage shown to the model in this run, whether from `retrieve_policy` or
  from the gate, which shows the clause it decided on; or
- the customer's own data: the ticket body, the account record (its dollar
  figures and seat count), past tickets, or a refund amount proposed in the run.

Restating the customer's own figures is not a policy claim. A sentence that
cites a retrieved clause is held to *that clause*, which is what clause-level
IDs exist for.

**Violations**, each a hallucinated-policy event:

| kind | meaning |
|---|---|
| `citation_not_retrieved` | cites a clause no retrieval in the run returned |
| `unsupported_fact` | a figure grounded nowhere |
| `miscited_fact` | a real, retrieved figure attributed to a cited clause that does not contain it |

**Coverage moved out of the gate.** ADR 0006's gate checked citations. The
check now lives in the guardrail, which runs before the gate, so that "without
guardrail" turns every claim-against-retrieval check off together. Otherwise
the ablation would compare "some coverage" with "more coverage". The gate is
left with the action-type half: the owner check, the bands, and pending
approval. A hallucinated reply is never parked for approval behind a held
refund.

**Block and detect.** `GUARDRAIL_MODE=block` (the default) stops the action and
escalates, and the escalation is filed with the violation as its reason.
`detect` logs the violation in the trace and on the result, and lets the action
proceed.

**The metric.** `hallucinated_policy_rate`: drafted replies with at least one
violation, over drafted replies. A blocked reply was still drafted, so it
counts in both modes. It is undefined, not zero, for an agent that drafts no
replies.

**The validation harness** (`python -m evals.guardrail_eval`) runs the same
check, with the same grounding function the loop uses, over
`evals/dataset/guardrail_cases.json`, and lists every disagreement with its case.

## Measurement

### The guardrail against labelled drafts

```
python -m evals.guardrail_eval
```

42 drafted replies against real tickets and real clause text. 16 are clean.
Two of those deliberately state true figures the checker cannot ground: a
per-seat price computed from the charge, and "over USD 50" paraphrasing
ESC-001.2's "USD 50.01". 20 carry checkable errors, and 6 carry errors with no
figure in them (`blind_spot`).

| | provisional, labels unreviewed |
|---|---|
| precision | 0.905 (19 of 21 flagged were truly wrong) |
| recall, all errors | 0.731 (19 of 26) |
| recall, checkable errors | 0.950 (19 of 20), an upper bound, see below |
| right kind on caught cases | 0.947 |

- **Both false positives are the two planted true-but-ungroundable figures.**
  The check cannot tell a computed or paraphrased figure from an invented one.
- **The 6 blind spots were all missed, by design:** "API access is only on
  Team", "you'll get a prorated credit", "you can go straight from Team to Free".
  Wrong claims with no figure are invisible to this check.
- **One checkable error was missed, and the reason generalises.** G-29 cites
  TRI-001.1 for "trials last 30 days". That is wrong, but the customer's own
  ticket says "it's been under 30 days", so the figure counts as the customer's.
  G-16 shows the same thing partially: the account's 5 seats masked a wrong
  "minimum of 5 seats". **When a policy figure happens to equal one of the
  customer's figures, the error is masked.**

**Why 0.950 is an upper bound.** The cases were written after the checker, by
the same author. A checkable case is one the author knew how to make
checkable. The checker was *not* tuned to the cases: G-29's miss was left
standing rather than fixed. But independence would need cases written by the
label owner, or drawn from the agent's own replies.

### The ablation, live

```
GUARDRAIL_MODE=block  python -m evals.run --agent src.agent.loop:LoopAgent   # x3
GUARDRAIL_MODE=detect python -m evals.run --agent src.agent.loop:LoopAgent   # x3
python -m evals.compare_runs block=... detect=... --trace var/trace/trace.jsonl
```

| | block (3 runs) | detect (3 runs) |
|---|---|---|
| hallucinated-policy rate | 0/12, 1/11, 0/11 | 0/12, 0/12, 0/11 |
| guardrail checks flagged | 1 of 54 | 0 of 56 |
| resolution rate | 0.364, 0.273, 0.318 (mean 0.318) | 0.364, 0.364, 0.318 (mean 0.348) |
| escalation precision | 0.700, 0.583, 0.636 (0.640) | 0.700, 0.700, 0.636 (0.679) |
| pending-approval rate | 0.136 ×3 | 0.136 ×3 |
| LLM cost per run | $0.0166 | $0.0163 |

"Checks" counts replies and refund reasons; the rate's denominator is drafted
replies.

**The one flag was a false positive.** On T-005 (gold `resolve`) the customer
wrote "we're onboarding two more engineers". The reply said "if you add the two
additional seats now, you will be charged right away", which is correct. "Two
engineers" does not ground "two seats", so the reply was blocked and the ticket
escalated. That is T-005's flip in the block arm.

**Read plainly: on this set, with this model, the guardrail caught no real
hallucinated figure, and its one intervention cost a correct resolution.**
Across 110 checks the agent never stated an ungrounded figure that the
checker's design could see. That is a null result for blocking, not evidence
that the agent never hallucinates:

- What the check cannot see, qualitative claims, is exactly what it cannot
  count. On the drafts that blind spot was 6 of 26 errors.
- Every run-level difference above is within run-to-run spread and inside each
  run's roughly ±0.18 resolution-rate interval.

This is the first measured case for CLAUDE.md's warning that the eval set is
too small. The guardrail's benefit, if there is one, needs a set where the
agent makes figure errors often enough to be blocked.

## Consequences

- `hallucinated_policy_rate` is reported by every `evals/run.py` run,
  alongside the other four metrics.
- The gate no longer checks citations; ADR 0006's status says so.
- `AgentResult` gains `policy_violations`.
- Blocking stays the default, because CLAUDE.md specifies it. The live
  measurement shows its cost (one false-positive escalation in three runs) and
  no benefit yet. Whether to keep blocking at this precision is a decision for
  the label owner, informed by the Session 8 set.

## Open

- **Label review** of `evals/dataset/guardrail_cases.json`. Every guardrail
  figure above is provisional until then.
- **Counts restated in other words** ("two engineers" → "two seats") are the
  observed false positive. A rule that lets a customer's head-count ground a
  seat count is plausible, but it was prompted by a held-out ticket. It should
  be designed and validated on the Session 8 synthetic set, not fitted to T-005.
- **Coincidence masking:** a policy figure equal to a customer figure passes
  (G-29, G-16). Holding cited sentences strictly to their clause would close it,
  but would also flag correct restatements such as G-19's "your 12 projects".
  That trade has not been measured.
- **An LLM judge for qualitative claims**, if the blind spot proves to matter.
  It would be measured as its own layer, never folded into the deterministic
  rate.
