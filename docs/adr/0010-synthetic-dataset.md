# 10. The synthetic dataset: hand-written specs, generated wording, reviewed labels

## Status

Accepted. Batch 1 (25 tickets) is generated and awaiting label review;
batches 2-6 follow once batch 1's review shows whether the specs need
changing.

## Context

CLAUDE.md asks for the 22 hand-written tickets to be expanded synthetically
to 150 or more, with a generator in `evals/generate/`, labels drafted by the
assistant and never auto-accepted, and four named targets the expansion must
fix (the Session 8 section): enough `pending_approval` tickets, non-owner
requests across several outcomes, a size where differences under about ±20
points can be resolved, and tickets whose correct outcome depends on history.
Generated tickets must cover the held-out set's range of writing styles,
deliberately sampling the messy end.

Four choices were made with the label owner before the code:

| question | choice | rejected |
|---|---|---|
| which non-owner tickets may be other than escalate | general questions answered by published policy alone | literal ESC-001.4, which contradicts target 2; amending the corpus, which changes what every retrieval number was measured on |
| who writes the text | an LLM, from hand-written specs | the assistant writing it, which is not reproducible from `evals/generate/`; templates, which come out too clean |
| size | 150 total: 22 held-out plus 128 synthetic | 200 |
| review | JSON batches of 25 with `"reviewed": false` | a review page in the browser |

## Decisions

**The spec is the source of truth; only the wording is generated.** Each
ticket starts as a hand-written spec in `scenarios.py`. It fixes the account,
who is asking, the facts the message must and must not contain, the outcome,
the expected tools, the relevant policies, and a rationale citing the
deciding clauses. That rationale becomes the drafted `label_note`. The writer
model fills in the words and nothing else. `checks.py` then rejects any
message that misses a required fact, contains a forbidden one, names a
clause or uses label wording. A failing message is rewritten up to three
times, and a spec that never passes stops the build. So a drafted label never
rests on the model's word that it said the right thing.

**The writer is gpt-4o, not the agent's gpt-4o-mini**, so the agent is not
graded on prose its own model wrote. The style examples in `writer.py` are
written for the generator, not copied from the held-out set, so the synthetic
set does not echo held-out wording. Temperature is 0.9, with a seed per
attempt. A rebuild does not reproduce the same wording, so the committed batch
file is the record and is never overwritten without `--force`.

**Writing styles are assigned per spec**, across four (clean, casual, messy,
buried), and at least 40% of a batch is messy or buried. Batch 1 is 48%. A
test enforces the floor.

**Ids, names and addresses are deterministic.** Synthetic customers are
`CUST-2NNN` and tickets `S-BBNN`, and names and companies come from fixed
pools by position. Requester addresses follow who is writing. An owner uses
the owner address. A colleague, member or assistant uses another person at
the same domain. A former employee writes from a personal address. An
impersonator uses the owner's name on a lookalike domain. So requester
identity stays structural, as the held-out set has it.

**Unreviewed batches never reach a number.** `load_synthetic` skips any batch
without `"reviewed": true`. `evals.run --set synthetic` refuses to run with no
reviewed batch, and `--set full` runs held-out then synthetic. A batch passes
the full loader validation at build time, so review is about whether the
labels are right, not whether they are well-formed.

**Non-owner ruling.** A non-owner asking a general question that published
policy answers alone, with no account action and no account data, gets an
answer. Everything else from a non-owner escalates under ESC-001.4. The loader
enforces this: a non-owner ticket labelled other than escalate may not expect
`issue_refund` and must carry a `label_note`. Recorded in `POLICY_NOTES.md`.

**History-dependent tickets must change the outcome.** The metrics score
outcomes, not reply wording, so a history fact that only changes what the
reply should say — a second trial refused, a grace extension already used —
is invisible to them. The specs use facts that move the outcome:

- *History in the fixtures:* a chargeback recorded only in the customer's
  past tickets turns an in-window refund (pending_approval) into escalate. A
  refund already issued for the same charge turns pending_approval into
  resolve.
- *Run order:* an earlier ticket in the same set, from the same customer,
  states the deciding fact (a bank dispute), and a later one asks for the
  refund without repeating it. Episodic memory stores the customer's words, so
  the fact reaches the later ticket whatever the agent did with the earlier
  one. The loader checks that the earlier ticket runs first, for the same
  customer.

With memory off, both kinds lose the fact, which is what makes the ablation
measurable.

**The agent has no clock.** It is never told the date, so a spec whose label
turns on a time window states the elapsed time in the message ("charged nine
days ago"). The eval world's today is 2026-08-24, the date the held-out set
implies, and account charge dates are consistent with it. A test enforces
this for every held refund.

## Found while writing the specs

- **REF-002.4 is not enforced by the gate.** A customer with two or more
  refunds in the past 12 months needs approval for any further refund,
  whatever the amount. The gate bands by amount only, so a $20 refund to such
  a customer would go through, against the label. Tickets testing it would
  measure a gap the agent cannot close. Batch 1 leaves it out. Enforcing it in
  the gate is a natural use of semantic memory's refund history, and a
  separate change.
- **Outcome-only scoring hides content errors.** A reply that grants a second
  trial, against TRI-001.5, scores the same as one that refuses it. That is
  why the history-dependent specs are restricted as above. It is also a
  limit on what resolution rate means generally.

## What it costs

- Every label still needs a human read. At 25 per batch, 128 tickets is
  about five review sessions.
- Hand-written specs keep generation honest, but they make the set only as
  varied as the specs are. The writer varies wording, not situations.
- gpt-4o writing costs about $0.04 per batch of 25.
