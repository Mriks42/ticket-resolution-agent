# 11. Refund history in the gate (REF-002.4), as account state

## Status

Accepted. Built; not yet measured, because no ticket in either set has a
refund history. The synthetic batches after batch 1 should add some.

## Context

REF-002.4: *"A customer who has received two or more refunds in the
preceding 12 months requires approval for any further refund, regardless of
amount."* The gate (ADR 0006) banded refunds by amount only, so a $20 refund
to such a customer executed under ESC-001.1, against the policy. ADR 0010
found the gap while writing synthetic specs and kept REF-002.4 tickets out of
batch 1, because the agent could not have got them right.

The rule needs a count of refunds the customer has received, which nothing
held in structured form. Two sources were weighed with the project owner:

| source | for | against |
|---|---|---|
| **account billing history** (chosen) | refunds are account state, like charges; the gate's input does not depend on the memory switch | a new record to keep, seeded in the fixtures |
| semantic memory's refund facts (ADR 0008) | already exists | with memory off the gate could not enforce REF-002.4, so the memory ablation would also measure a different gate: the kind of confound ADR 0008 had just removed |

## Decisions

**Billing history is account state.** `Customer` gains `refunds` (date,
amount, ticket), seeded from the fixtures. `BillingHistory`
(`src/tools/billing.py`) holds them, one per `ToolContext` like memory, so one
eval run's refunds never reach the next. `lookup_customer` shows them under
`refunds` when there are any. The key is absent otherwise, so held-out
account records read exactly as before. Memory off does not touch them.

**Only refunds received count.** REF-002.4 says *received*. A refund the
gate executes is added to the history, with today's date and the ticket. A
refund held for approval is not added, because it has not been received. Within
one ticket, refunds already executed earlier in it count as well.

**The gate counts from its own lookup.** The owner check already looks the
account up through the audited read server, never trusting anything the
model said. The same lookup now supplies the refunds. The count covers
refunds dated within the 365 days up to and including today. A record with
an unreadable date counts, failing closed, as the owner check does.

**The rule only ever holds.** If the refund would have executed by amount
(ESC-001.1) and the customer has two or more refunds in the window, it is
held for a support manager under REF-002.4. A refund already in a higher band
keeps that band's approver and clause. History never lowers a band, and the
owner check still comes first.

**Refund amounts in the history ground replies** as the customer's own
figures, like the account's charges (ADR 0007).

**The loop has a calendar clock**, injectable for tests, used for the
12-month window and for memory's close dates.

## What it costs

- **Two clocks disagree.** The eval world's today is 2026-08-24 (ADR 0010),
  but the gate uses the real date. Refunds the agent executes in a run are
  dated with the real date, and fixture refunds with eval-world dates. A
  12-month window is long enough that fixtures dated within the eval world's
  preceding year stay inside it for now. A fixture refund near the window's
  edge would land on different sides under the two dates. Synthetic specs
  should keep their refunds well inside it, or the eval world's date should
  become configuration.
- `run_finished` and the gate trace carry `prior_refunds`. Nothing reports
  REF-002.4 holds separately from band holds except the gate's `rule`.
- The approver for a REF-002.4 hold is assumed to be a support manager. The
  clause says "requires approval" without naming who; the support manager is
  the lowest approver ESC-001 names.
