# 3. Embedding-based retrieval over pgvector, and reordering it before the agent loop

## Status

Accepted

## Context

Session 3 shipped `retrieve_policy` on a lexical TF-IDF index
(`src/tools/lexical.py`), chosen because it needs no dependencies and returns
an honest empty list when nothing matches. Measuring it against the labels
already in the dataset showed it is not good enough to build the agent loop on
top of.

This ADR records that measurement and the resulting decision to move retrieval
ahead of the agent loop in the session order.

### How these numbers were produced

```
python -m evals.retrieval_baseline
```

Each ticket's `body` verbatim as the query against the 104-clause corpus;
retrieved clause IDs rolled up to their `policy_id` and compared against that
ticket's `relevant_policy_ids`. The rollup is necessary because labels are
document-level (`POL-REF-001`) while retrieval is clause-level (`REF-001.3`),
and it flatters the retriever slightly: returning any clause of a document
counts as retrieving it, even the wrong clause.

### What the measurement found

Recall of the labeled-relevant policy documents, ticket body as query, with the
context each k costs:

| k | mean recall | tickets with all relevant docs | ESC-001 retrieved | chars | ~tokens |
|---|---|---|---|---|---|
| 3 | 0.500 | 5/22 | 1/11 | 280 | 70 |
| 5 | 0.530 | 5/22 | 1/11 | 481 | 120 |
| 7 | 0.561 | 5/22 | 1/11 | 673 | 168 |
| 8 | 0.652 | 8/22 | 2/11 | 793 | 198 |
| 10 | 0.716 | 10/22 | 3/11 | 994 | 249 |
| 12 | 0.727 | 10/22 | 3/11 | 1206 | 302 |
| 15 | 0.727 | 10/22 | 3/11 | 1527 | 382 |
| 20 | 0.822 | 13/22 | 4/11 | 2036 | 509 |

At the `limit=5` this shipped with, the agent would on average see about half
the policy it needs, and on 17 of 22 tickets would be missing at least one
relevant document.

### Correction: the original T-016 finding was wrong

The session 3 report claimed the planted REF-001.3 / CAN-002.2 conflict was
unreachable from realistic phrasing. **That was wrong, and it is recorded here
rather than quietly fixed, because the reordering below was originally
justified by it.** The claim came from testing a paraphrase written by hand —
*"annual auto renewed 10 days ago want money back"* — instead of the ticket.
Against T-016's actual body:

```
REF-001.3  rank 1
CAN-002.2  rank 6
```

Both sides are reachable. CAN-002.2 was simply falling outside `limit=5`. The
lesson generalises past this one case: a retriever must be evaluated on the
corpus's real queries, which is now what `evals/retrieval_baseline.py` does and
what the regression test in `tests/test_tools.py` pins.

This is a real finding in its own right, and a cheaper one than a new backend —
see the decision on the default limit below. It is not, however, a reason to
leave retrieval as it is; the measurement that corrected it also surfaced a
larger problem.

### The actual dominant failure: POL-ESC-001 is not retrieved

`POL-ESC-001` (approval thresholds and mandatory escalation) is labeled
relevant on **11 of 22 tickets** — it is the document that decides whether a
ticket is `resolve` or `pending_approval`. It is retrieved on:

| k | tickets where ESC-001 was retrieved |
|---|---|
| 5 | 1 / 11 |
| 10 | 3 / 11 |
| 20 | 4 / 11 |

For the three `pending_approval` tickets, the first ESC-001 clause ranks (out
of the clauses scoring above zero at all, so "last" is checkable):

```
T-017   rank 75 of 75   (last scoring clause)
T-021   rank 36 of 73
T-022   rank 28 of 71
```

This is worse than it looks. The `pending_approval` base is already n=3 (see
the Session 8 targets in CLAUDE.md). If the one document that determines that
outcome is never retrieved, pending-approval rate measures retrieval failure
and nothing else, on all three tickets.

**This is not a similarity problem.** The index reaches ESC-001 fine when the
query names it — `"refund approval threshold manager"` returns ESC-001.2,
ESC-001.3, ESC-001.1 as the top three. The problem is that a customer never
writes that. T-021 says *"I'd like that refunded and the account closed"*; the
clause that governs it says *"A refund between USD 50.01 and USD 500 requires
support manager approval."* The link between them is `$176 falls in a band`, a
numeric threshold lookup. No amount of similarity between the ticket text and
the clause text will produce it, because the relation is not similarity.

So the corpus contains two distinct failure classes, and they need different
fixes:

- **(A) Recall and vocabulary.** The right clause exists, shares meaning with
  the query, and is ranked too low or missed for want of shared words. T-012
  (*"we stopped using this back in March"*, recall 0.00 at k=10, missing
  POL-CAN-002 and POL-ESC-001) is the clearest case.
- **(B) Query-intent mismatch.** The governing clause has no lexical *or*
  semantic relation to the ticket text; it is reached by applying a rule to a
  value in the ticket. All of ESC-001's threshold clauses sit here.

## Options considered

**Keep lexical, raise the default `limit`.** Cheapest thing that works, and it
does fix T-016. It moves mean recall from 0.530 to 0.716 — still one in three
relevant documents unseen, and it barely touches class (B): ESC-001 goes from
1/11 to 3/11. It also enlarges every prompt, and a larger k with a weak ranker
means more irrelevant passages the guardrail must reason around. Not sufficient
on its own, but strictly worth doing, and adopted below.

**Embedding retrieval over pgvector.** Already in the stack per CLAUDE.md, so
no new dependency argument to make. Directly targets class (A). Does nothing
for class (B) by itself.

**Hybrid (lexical + dense, fused).** Keeps exact-term matching for clause IDs,
amounts and plan names, where lexical is strong and dense is weak, while adding
paraphrase robustness. Most likely landing spot, but it should be chosen on
measured recall rather than asserted here.

## Decision

1. **Retrieval moves ahead of the agent loop in the session order.** This is
   an empirical decision, not a preference. With ESC-001 retrieved on 1 of 11
   tickets at the shipped default, an agent loop built now would be tuned
   against a signal dominated by retrieval failure — and the tuning would
   silently bake that in.

2. **Raise the default `limit` from 5 to 10, now.** Chosen off the sweep
   rather than as a round number. Recall climbs in two steps — k=8 (+0.091,
   and three more tickets reach full recall) and k=10 (+0.064, two more) —
   and then goes flat: 0.716 at k=10 against 0.727 at *both* k=12 and k=15.
   k=10 is the knee, the last value that buys anything before diminishing
   returns.

   **What it buys:** mean recall 0.530 → 0.716; tickets with every relevant
   document retrieved 5/22 → 10/22; T-016's conflict becomes visible, since
   CAN-002.2 at rank 6 now falls inside the window.

   **What it costs:** mean returned passage text goes from 481 to 994
   characters per ticket, roughly 120 to 250 tokens. That is a real but small
   addition to a per-ticket prompt, and it is the cheaper half of the trade —
   the more material cost is that a weak ranker at k=10 puts more irrelevant
   passages in front of the guardrail, which has to keep refusing to cite
   them. If hallucinated-policy rate rises when the loop lands, this is the
   first thing to re-examine.

   k=20 would reach 0.822, but it doubles context again for a ranker we are
   about to replace, so it is not worth buying twice.

3. **Adopt embedding-based retrieval over pgvector** as the primary backend
   for `retrieve_policy`.

4. **Keep the lexical index.** It is the baseline the new backend has to beat,
   and this project does not get to claim an improvement it has not measured.
   `src/tools/lexical.py` stays, selectable by config.

5. **`evals/retrieval_baseline.py` is the regression check.** Recall@k against
   `relevant_policy_ids` needs no agent loop, no LLM and no cost, so it runs on
   every retrieval change. It produces every number in this ADR.

6. **Add retrieval backend as an ablation axis** (lexical / dense / hybrid),
   alongside the three ablations CLAUDE.md already requires.

### Class (B) is a gating rule, not a retrieval target

Splitting this out because it is the thing most likely to be mistaken for a
retrieval bug and chased with a better index.

CLAUDE.md already specifies HITL gating as deterministic, on **action type**
combined with **retrieval coverage**. The approval bands in ESC-001.1–.3 are
the *action-type* half of that rule. "Refund of $176" falls in the
50.01–500 band and therefore requires manager approval — that is a comparison
against a number the agent already holds, decided in code, before any passage
is retrieved.

So ESC-001's thresholds are **not** something `retrieve_policy` is expected to
supply, and the 3/11 figure above is not a target to optimise. Retrieval's job
for these tickets is the *coverage* half — did the reply's claims land on
passages actually retrieved — and that is a different question from whether the
gate fires.

Two consequences follow:

- Session 6 must not treat the ESC-001 recall number as a retrieval defect to
  fix. Implementing the band comparison in the gate is the fix. A dense index
  that happened to raise 3/11 to 8/11 would be measuring nothing useful, since
  the gate would not be consulting those passages to decide.
- The gate still needs the clause text to *explain* itself in the reply, and
  citing ESC-001.2 requires having retrieved ESC-001.2. So the agent should
  issue a targeted policy query once the amount is known, rather than hoping
  the ticket body reaches the clause. That is a loop-design point, and it
  belongs to the session that builds the gate.

This keeps the two halves of the gating rule in their proper places: the
threshold is arithmetic, and retrieval is evidence.

## Consequences

### What pgvector is expected to fix

- Class (A) recall: paraphrase and vocabulary mismatch, where the customer's
  words and the clause's words differ but the meaning does not. T-012, T-022,
  T-005 and T-008 are the concrete cases.
- Higher recall at low k, which keeps prompts small and reduces the number of
  irrelevant passages the guardrail has to survive.

### What it will not fix, and what it may break

These are predictions, flagged as such. Each needs to be checked against the
retrieval eval rather than assumed:

- **Numeric bands (class B).** Embeddings are weak at magnitude comparison;
  `$176` and `$1,584` embed similarly while falling in different approval
  bands. Expect no improvement on ESC-001 — and per "Class (B) is a gating
  rule" above, do not chase one. The band comparison belongs in the gate.
- **Negation.** "refundable" and "non-refundable" are close in embedding
  space. That may help surface both sides of the REF-001.3 / CAN-002.2
  conflict, and equally may make the agent unable to tell which one it is
  looking at. The planted conflict is the test for this.
- **The empty result disappears.** This is the sharpest regression risk.
  Lexical returns nothing when nothing matches, and the deliberate corpus gaps
  in `POLICY_NOTES.md` depend on that: T-017, T-018 and the four unlabelled
  gaps are traps that only work if "no relevant policy" is expressible. Dense
  retrieval always returns a nearest neighbour. A similarity floor is
  therefore mandatory, not optional, and the floor has to be calibrated
  against the gap tickets — otherwise the retrieval guardrail loses the signal
  it fires on and the hallucinated-policy metric degrades.
- **Exact-ID lookup gets worse.** Fetching `REF-001.3` by name is something
  lexical does exactly and dense does approximately. Another argument for
  hybrid.
- **Eval comparability across model versions.** Rankings depend on the
  embedding model, so the model name and version become part of the eval
  record, and numbers produced under different models are not comparable.
  Pin it and log it.
- **Cost and latency.** An embedding call per query, plus indexing the corpus,
  lands in cost-per-resolved-ticket and p50/p95 latency. Currently retrieval
  is free and sub-millisecond; that will no longer be true, and the budget
  numbers in CLAUDE.md were not set with it in mind.

### What changed under this ADR

Applied now, with the ADR:

- `DEFAULT_RETRIEVAL_LIMIT` 5 → 10 in `src/tools/catalog.py`, per decision 2.
- `evals/retrieval_baseline.py`, which produces every number above.
- A regression test pinning that T-016's conflict is reachable at the default
  limit, queried with the ticket's real body.

Deferred to the retrieval session: the pgvector backend itself, the similarity
floor, and the backend ablation axis.

### On the session order itself

The agent loop, the HITL gate and the guardrail all shift one session later.
The tool and MCP layer is otherwise untouched: `retrieve_policy` keeps its
signature and its clause-level IDs, and the backend swap happens behind
`src/tools/policy.py`. The catalog, the scopes and the audit log are unaffected.
