# 9. Cost and latency per resolved ticket

## Status

Accepted. First measured 2026-10-26 (*Measured*). Reports from before this
ADR carry no per-ticket cost or time and show `n/a`.

## Context

CLAUDE.md requires two metrics from `evals/run.py` beside the rates: *cost
per resolved ticket — USD, traced across all steps in the run* and *p50 / p95
latency per resolved ticket*, both with bootstrap confidence intervals.
Before this ADR the loop traced LLM cost and wall clock per run, but nothing
reached the report, and the embedding calls made by dense retrieval and by
memory (ADR 0008) were counted nowhere.

Each metric leaves a definition open. These were settled by the assistant
under a standing "do what you think is best" from the project owner, and are
recorded here so they can be overturned.

## Decisions

**"Resolved" means correctly resolved** — the numerator of resolution rate:
the agent resolved and the label says resolve. A wrong resolution is not a
resolution the project wants to price or time.

**Cost per resolved ticket is all spend over correct resolutions:**
`sum(cost of every ticket) / count(correctly resolved)`. Spend on tickets
that escalated, went to approval or resolved wrongly is part of what each
correct resolution cost. It is a ratio of sums, so it uses the same
ticket-level bootstrap as every other rate. It is undefined, not infinite,
when nothing is resolved.

- *Rejected:* the mean cost of the resolved tickets alone. It is cheaper to
  read and flatters an agent that spends heavily on the tickets it fails.

**A ticket's cost is everything its run paid for**:

- LLM tokens as the provider counted them, at the list price in
  `src/agent/pricing.py`. Cached input is charged at the full rate, so this
  overstates slightly, as before.
- Embedding tokens the provider counted while that ticket ran, at list
  price: query embeddings for `retrieve_policy` and `search_past_tickets`,
  and the episode memory writes at close. An `EmbeddingMeter` reads the
  embedders' cumulative token counts before and after each run. The
  difference is the run's, which is exact while runs are sequential, as an
  eval's are. Embedding a corpus or seeding memory happens before the first
  ticket and is charged to none.
- Text already in the embedding cache costs nothing, because nothing was
  paid. So a rerun can cost a little less than a cold run of the same
  configuration. Charging cache hits as if they were calls would make the
  number reproducible, but it would report spend that never happened.

Embedding spend also counts against the eval's spend cap. The trace keeps
the two apart: `run_finished.cost_usd` stays LLM only, as `compare_runs` has
always labelled it, and `embedding_cost_usd` sits beside it. The result's
`cost_usd` is the total. An agent without a price on record reports cost as
unknown (`None`), not zero.

**Latency is wall clock per ticket**, from the run's start to its terminal
state, including the close's memory write. It covers model calls, tool calls
and embedding calls: what a customer would wait. p50 and p95 are taken over
correctly resolved tickets with linear interpolation, and bootstrapped by
resampling tickets and recomputing the percentile over the resolved ones in
each resample.

**Not reported is not zero.** An agent whose results carry no cost or no
time — the stubs, and any report from before this ADR — gets `n/a` with the
reason in the caveat, never a number.

## Measured

Two runs at the default configuration (gpt-4o-mini, dense retrieval,
guardrail blocking, memory on), on the 22 held-out tickets:

```sh
uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/cost_latency_run1.json
uv run python -m evals.run --agent src.agent.loop:LoopAgent --out evals/report/cost_latency_run2.json
uv run python -m evals.compare_runs --trace var/trace/trace.jsonl \
    run=evals/report/cost_latency_run1.json run=evals/report/cost_latency_run2.json
```

| | run 1 | run 2 |
|---|---|---|
| cost per resolved ticket, USD [95% CI] | 0.00235 [0.00136, 0.00590] | 0.00254 [0.00144, 0.00648] |
| p50 latency per resolved ticket, s [95% CI] | 3.76 [3.28, 7.78] | 3.51 [3.17, 4.68] |
| p95 latency per resolved ticket, s [95% CI] | 10.63 [3.76, 12.08] | 4.77 [3.51, 4.81] |
| correctly resolved tickets | 7 | 7 |

- **Embeddings are a rounding error in cost.** Over the 44 tickets they came
  to $0.000016, against $0.0343 of LLM spend, about 0.05%, because most
  query text was already in the cache. A cold cache would cost more, but the
  same order of magnitude.
- **p95 moved from 10.6 s to 4.8 s between identical runs.** With seven
  resolved tickets it is the slowest one or two, so one slow model response
  sets it, as predicted above. Read p50.
- Both cost intervals span more than a factor of four. That is seven
  resolutions in the denominator, not noise in the price.

## What it costs

- With about ten correct resolutions in the 22-ticket set, p95 is close to
  the slowest resolved ticket, and its interval is wide. The caveat says so.
  It becomes a percentile in the ordinary sense only at Session 8 sizes.
- Latency depends on the provider's load on the day, and on whether Docker
  and Postgres are warm. Two runs of one configuration on different days are
  not comparable on latency the way they are on rates.
- Per-run embedding attribution assumes sequential runs. A concurrent runner
  would need per-call attribution instead.
- Only `text-embedding-3-small` has a price on record; the meter refuses to
  build for an embedding model without one, as the spend guard does for chat
  models.
