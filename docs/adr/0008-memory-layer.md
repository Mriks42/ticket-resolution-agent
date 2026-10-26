# 8. The memory layer: what "off" removes, and what the agent may write

## Status

Accepted. Built and measured on the held-out set, 2026-10-26. What that can
show is limited by construction (*What this can measure*). The first
measurement exposed a confound in the off arm; the off arm was changed and
measured again (*Measured*).

## Context

CLAUDE.md asks for two memory layers kept separate — **episodic** (past
tickets and their resolutions for a customer, retrieved by embedding
similarity over pgvector) and **semantic** (durable facts about a customer,
written only on ticket close) — and for memory to be switchable off, because
the headline ablation is resolution rate with memory against without.

Before this ADR, history already reached the agent through the fixtures:
`search_past_tickets` searched `past_tickets` in `customers.json` lexically,
and `lookup_customer` returned the same `past_tickets` inline in the account
record. Nothing was ever written.

Three findings shaped the design before any code:

1. **On the held-out set the ablation is near-null by construction.** There
   are 22 tickets from 22 different customers, so anything written when one
   ticket closes is never read by another. Only three customers have history
   (T-009, T-015, T-020), and on all three the ticket text alone decides the
   outcome: T-009 says a chargeback is filed (ESC-001), T-015 reports a double
   charge, T-020 is a decline under SLA-001.5 whatever the history says. The
   largest possible difference is 3 of 22, about 14 points, inside the ±20
   point interval CLAUDE.md warns about.
2. **Hiding the search tool does not make a run stateless.** `lookup_customer`
   carries `past_tickets` too.
3. **History grounded the guardrail across customers.** A search with no
   `customer_id` searched every customer, and the loop adds search results to
   the text the guardrail treats as the customer's own records (ADR 0007). A
   figure from another customer's ticket could ground a reply: "$88" from
   CUST-1560's history let a reply to CUST-1042 resolve. No recorded run did
   this — all 24 audited searches passed the ticket's own customer — but
   nothing prevented it.

Six choices were made with the label owner before the code:

| question | choice | rejected |
|---|---|---|
| what "off" removes | all history: what the search returns and `past_tickets` (and remembered facts) in the account. First built by hiding the tool and the field; changed to emptying them after the measurement below | agent-written memory only, which leaves both arms identical on the 22; the search tool only, which leaks history through `lookup_customer` |
| how facts are extracted | deterministically, from the run's own records | an LLM at close, whose invented fact would be read back as the customer's own record |
| search scope | the ticket's own customer only | cross-customer precedent, which needs a second grounding class to be safe |
| when a run writes | at every terminal state | RESOLVE only, which leaves an escalated ticket with no record, since in an eval no human ever closes it |
| what an agent-written past ticket holds | the customer's words and a fixed-template outcome | the reply text |
| episodic backend | follows `RETRIEVAL_BACKEND` | always pgvector |

## Decisions

**The switch** is `AgentConfig.memory`, read from `MEMORY_ENABLED` (default
`true`, the name `.env.example` already reserved), and recorded in every
`run_started` trace event. Off, the loop:

- offers the same tools, but a `search_past_tickets` result reaches the model
  with no matches;
- empties `past_tickets` in the `lookup_customer` result, keeping the field
  where it was, and drops `memory`, before the model or the guardrail sees
  either;
- writes nothing at close.

So for a customer with no history, the two arms see byte-identical input, and
a test holds the loop to that. The first version hid the tool and removed the
field instead. That changed every ticket's input, and the change alone moved
outcomes (*Measured*). The searches still run and are audited; only what the
model and the guardrail see is emptied.

The account itself — plan, seats, billing cycle, last charge — is current
state rather than memory, and stays in both arms. On is what every run did
before this ADR, so earlier reports remain comparable to the on arm.

**Search is pinned to the ticket's customer.** The loop sets `customer_id` on
every `search_past_tickets` call to the ticket's customer, whether the model
omitted it or named someone else. The catalog is unchanged: an MCP caller
other than the loop can still search across customers, and gets no grounding
from it because only the loop grounds anything.

**Episodic memory** (`src/memory/episodic.py`) holds past tickets: customer,
ticket id, date, summary, resolution. It is seeded from the fixtures'
`past_tickets`, which are the eval world's prior history. It backs both
`search_past_tickets` (ranked by similarity to the query) and `past_tickets`
in `lookup_customer` (the whole history, oldest first), so the two cannot
disagree.

- With `RETRIEVAL_BACKEND=lexical`, the code default, it is kept in the
  process with the TF-IDF index the search used before. Tests and CI need no
  Postgres.
- With `dense` or `hybrid`, which is what `.env` uses for evals, it is a
  pgvector table searched by cosine similarity, reusing the embedding cache
  of ADR 0004, so reruns make no new embedding calls for text already seen.

**No similarity floor on history.** ADR 0004's floor exists so that "no
policy covers this" stays expressible. For history the empty result already
means something else — "this customer has never written in" — and a floor
would make real history vanish whenever the query was phrased differently.
Search returns the customer's history ranked, up to the limit.

**Each eval run has its own memory.** `ToolContext.build` gives every agent
instance — one eval run — a fresh namespace, seeded from the fixtures. Writes
accumulate in ticket order within the run and never reach another run, so a
rerun starts from the same state. In-process this is automatic; in Postgres
each row carries the namespace. A persistent production namespace is not
built: there is no production path yet.

**When a run writes.** At the end of every run, after its terminal state is
reached and any system escalation is filed, never mid-run — RESOLVE, ESCALATE,
PENDING_APPROVAL and FAILED alike. A repeat contact needs to know that the
last one was escalated. A held refund is recorded as held, not issued. The
write is traced (`memory_written`), not audited: it is not a tool call and
not visible to the customer. A failed write raises out of `run()`, like the
spend cap, so an eval whose memory arm silently lost its writes cannot report
numbers.

**What an agent-written past ticket holds.** The summary is the ticket body,
which is the customer's own words and already grounding in the run that wrote
it. The resolution is a fixed template built from the run's records: the
terminal state, refunds with amount and status, and the clause ids cited.
There is no reply text and no escalation reason. The guardrail treats history
as the customer's own records, so the agent's earlier prose would ground its
later claims: a figure invented in one reply, sent in detect mode, would
ground the same figure in the next reply without any retrieval.

**Semantic memory** (`src/memory/semantic.py`) is one record of facts per
customer, folded on each close by a pure function of the run's records:

- `tickets_closed`;
- `refunds` — each with ticket, amount, status (`issued` or `held`) and date;
- `escalations` — each with ticket, date, and the clause ids it rested on;
- `policies_cited` — how many closed tickets cited each policy document, and
  `recurring`, the documents cited at two or more. This is a deterministic
  stand-in for "recurring issue". POL-ESC-001 is left out because it governs
  procedure rather than describing a subject.

Plan tier is left out: the account record is the authority on it, and a
remembered copy can only go stale. The facts appear under `memory` in the
`lookup_customer` result when there are any, and the key is absent when there
are none. So on a customer with no closes the on arm sees exactly what it saw
before this ADR. They are not grounding: `account_grounding` reads only
top-level figures and `past_tickets`.

## What this can measure

On the 22 held-out tickets, memory can change the input on three tickets and,
on the labels, the right outcome on none. An ablation run there tests that the
switch works and costs nothing, not that memory helps. Writes are exercised
only by the tests, because no customer appears twice.

The measurable version needs tickets whose correct outcome depends on
history: a second trial request (TRI-001.5), a refund already issued for the
same charge, or a repeat customer whose earlier ticket in the run establishes
the deciding fact. Added to CLAUDE.md as the fourth Session 8 expansion
target. Until that set exists, no README claim about memory is supportable.

## Measured

Three runs per arm, alternated, on the 22 held-out tickets: gpt-4o-mini,
dense retrieval, guardrail blocking. Reports are under `evals/report/`
(gitignored), and the run traces are in `var/trace/`.

```sh
MEMORY_ENABLED=true  uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/memory_on_run1.json
MEMORY_ENABLED=false uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/memory_off_run1.json
# ... run2, run3 likewise, then
uv run python -m evals.compare_runs --trace var/trace/trace.jsonl \
    on=evals/report/memory_on_run1.json on=... off=evals/report/memory_off_run1.json off=...
```

| | on | off |
|---|---|---|
| resolution rate, per run | 0.318, 0.364, 0.364 | 0.364, 0.364, 0.364 |
| pending-approval rate, per run | 0.136, 0.136, 0.136 | 0.045, 0.136, 0.091 |
| escalation precision, mean | 0.679 | 0.657 |
| tool-call precision, mean | 0.776 | 0.781 |
| hallucinated-policy rate | 0 in every run | 0 in every run |
| tickets whose outcome differs between runs | 1 of 22 | 4 of 22 |
| LLM cost per run | $0.0172 | $0.0153 |

Means across runs are not confidence intervals; each report carries its own
bootstrap CI. LLM cost excludes the embedding calls memory makes.

- **The three tickets with history came out the same in all six runs**:
  T-009 escalate, T-015 pending_approval, T-020 escalate. Memory content
  changed no outcome, as predicted. The on arm wrote 66 episodes, one per run,
  and none was ever read back, since no customer appears twice.
- **The switch moved outcomes on tickets with no history.** T-022, with no
  history, is pending_approval in all three on runs and resolve in all three
  off runs. Off, the model's first search is a vaguer "refund policy", and it
  issues a $50 "goodwill" refund that executes under ESC-001.1 instead of the
  $264 refund held under ESC-001.2. The label says pending_approval. T-011,
  T-016, T-017 and T-021 also vary between off runs. On these tickets the
  two arms' inputs differ only in what the switch removes from every ticket:
  one tool from the menu, and the empty `past_tickets` field from the account.
  So the off arm as built measures history *and* a changed interface, and the
  interface change alone moved one ticket three times out of three.

**The off arm was changed to keep the interface** — the tool still offered,
history and facts emptied in place — so the two arms' inputs are identical
except where history exists, and any difference is memory's. Three new off
runs (`memory_off_v2_run1..3.json`) against the same three on runs:

| | on | off, second version |
|---|---|---|
| resolution rate, per run | 0.318, 0.364, 0.364 | 0.318, 0.318, 0.364 |
| pending-approval rate, per run | 0.136, 0.136, 0.136 | 0.182, 0.136, 0.136 |
| escalation precision, mean | 0.679 | 0.679 |
| tool-call precision, mean | 0.776 | 0.784 |
| hallucinated-policy rate | 0 in every run | 0 in every run |
| tickets whose outcome differs between runs | 1 of 22 | 2 of 22 |
| LLM cost per run | $0.0172 | $0.0178 |

T-022 is now pending_approval in all six runs. The arms' majority outcomes
agree on every ticket except T-008 and T-021, and both of those also
disagree between runs of the same arm. The three tickets with history come
out the same as before. This is the null result the held-out set allows: the
switch works and changes nothing else. Whether memory helps needs the
history-dependent tickets of Session 8.

**A leak found afterwards.** Before a refund, the gate looks the account up
itself to verify the owner. That lookup was absorbed into guardrail grounding
with its history, even with memory off. It was fixed by sending it through the
same emptying as the model's lookups. In the second off arm, the only history
ticket that reached the gate was T-015. Its held replies quote $58 (in the
message) and $132 (the account's last charge), neither of which needed the
history. So no outcome above depended on the leak.

## What it costs

- The lexical in-process store and the pgvector store are two
  implementations of one contract to keep in step. The contract tests run
  against both where Postgres is up.
- Under `dense`, every eval run inserts its own copy of the seed history and
  its writes into Postgres and never deletes them. That is a few dozen rows
  per run, and they double as a record of what memory each run had, but
  nothing prunes them.
- Episodes store customer prose, the ticket body, in Postgres. ADR 0004 kept
  query text out of the embedding cache for exactly that reason. The episode
  table is the one place it is deliberate, because remembering what a customer
  said is what episodic memory is.
- Pinning the search means the model cannot look up a related customer — a
  colleague on the same account, say. Nothing in the dataset needs that yet.
