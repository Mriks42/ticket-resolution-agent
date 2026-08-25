# evals

Every number that ends up in the top-level README comes from here.

```bash
python -m evals.run                                    # stub baseline
python -m evals.run --agent pkg.mod:MyAgent --out evals/report/mine.json
```

`--agent` takes a `module:ClassName` implementing the `Agent` protocol in
[`schema.py`](schema.py): `name`, and `async run(ticket) -> AgentResult`.

## What's here

| Path | What it is |
|---|---|
| `dataset/policies/` | 20 policy documents, stable `POL-XXX-NNN` ids, clause-level ids for citation |
| `dataset/tickets.json` | 22 hand-written labeled tickets |
| `dataset/customers.json` | Account fixtures behind `lookup_customer` and `search_past_tickets` |
| `dataset/POLICY_NOTES.md` | Reviewer notes: planted conflict, deliberate gaps, label rulings. Not indexed |
| `run.py` | Runs an agent, prints metrics, writes the JSON report |
| `metrics.py` | Metric definitions and the percentile bootstrap |
| `agents/stub.py` | Always-escalate baseline: the floor to measure against |

Gold outcome distribution is 10 resolve / 9 escalate / 3 pending_approval.
`loader.py` refuses to load a dataset with duplicate ticket ids, policy ids
absent from the corpus, customers without fixtures, or a ticket expecting
`search_past_tickets` from a customer with no history.

## Known limitations

Read these before comparing two agents. At the current sample size most
differences are not measurable, and the harness is built to say so rather than
to present noise as a result.

- **22 tickets is not enough to compare agents.** The dataset is an eighth of
  the 150–200 the project calls for. A 95% CI on a rate over 22 tickets spans
  roughly ±20 points, so two agents differing by less than that are
  indistinguishable here. Treat any gap smaller than the CI width as unproven,
  not as a small win.
- **Pending-approval rate has n=3 and is not usable for comparison.** With
  three gold `pending_approval` tickets the metric moves in ~33-point steps —
  one ticket flipping is a third of the range. It is reported for completeness
  and to catch an agent that never requests approval at all. Do not read a
  difference in it as signal. Session 8's dataset expansion has to raise this
  base meaningfully; the other three metrics scale with the ticket count, but
  this one needs `pending_approval` tickets specifically.
- **A zero-width CI means a constant, not a certainty.** When an agent scores 0
  on every ticket, every bootstrap resample returns 0 and the interval collapses
  to `[0.000, 0.000]`. The run output flags this. It is the absence of
  variation, not a precise estimate.
- **Tool-call precision is name-level only.** It matches tool names against the
  label's `expected_tools` and ignores argument correctness, call ordering, and
  repeated calls, so an agent that loops on a correct tool still scores 1.0.
  Argument correctness, ordering, and loop detection become separate metrics
  once the real agent lands (see the TODO in `metrics.py`). Recall is not
  measured at all.
- **One label sits below high confidence.** T-017 is `medium`; its `label_note`
  records why. Every metric inherits that uncertainty and no CI accounts for it,
  because bootstrap resamples tickets, not label correctness.
- **Escalation precision is undefined, not zero, when an agent escalates
  nothing.** The harness reports `n/a` and says why. An agent that escalates
  everything scores exactly the gold escalate share, which is why this metric is
  meaningless without resolution rate beside it.

## Labels

Each ticket carries the correct outcome, the tools a competent human would have
called, and the policy ids relevant to it. `confidence` plus `label_note` record
how settled the label is: `label_note` holds the reasoning behind a reviewed
decision, or the open question on one still in doubt. Anything below `high`
confidence must carry a note — `schema.py` enforces it.
