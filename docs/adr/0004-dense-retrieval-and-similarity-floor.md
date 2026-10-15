# 4. Dense retrieval over pgvector, with a calibrated similarity floor

## Status

Accepted. The floor is settled at 0.30 (see "The floor"). One item is open:
the negative-probe set has had an assistant evidence pass but no human review,
so every figure marked † is provisional.

Corrects one premise of ADR 0003: see "The gap tickets cannot be reached by a
floor".

## Context

ADR 0003 adopted embedding retrieval over pgvector and deferred building it to
this session, along with the mandatory similarity floor and the backend
ablation axis. It predicted what dense retrieval would and would not fix. This
ADR records what was built, what was measured, and which predictions held.

Choices made before measuring, and why:

- **`text-embedding-3-small`** (1536-d). Same provider as the `gpt-4o-mini`
  default, so no new credentials. The model name is config
  (`EMBEDDING_MODEL`) and is written into every report.
- **One new dependency, `psycopg[binary]`.** Vectors cross the wire as pgvector
  text literals, so the `pgvector` Python adapter is not needed. The embedding
  call is one POST and goes over stdlib `urllib`, for the same reason ADR 0002
  hand-rolls the MCP server.
- **Exact scan, no HNSW.** At 104 clauses a sequential scan is exact and
  costs microseconds; an approximate index would add a second source of
  ranking variance to an eval trying to measure one.
- **Every vector cached in Postgres**, keyed by model and SHA-256 of the exact
  text. OpenAI publishes no dated snapshots of its embedding models, so the
  model name alone does not pin a ranking; the stored vectors do. Query rows
  keep only the hash, never ticket text.
- **The floor rule was fixed in code before any dense number existed**: the
  highest floor, on a 0.01 grid, at which mean recall@10 over the labeled
  tickets is unchanged from the unfloored run. Probe rejection is reported as
  the consequence, not optimised.

### How these numbers were produced

```
docker compose up -d
python -m evals.retrieval_baseline --backend lexical dense --calibrate-floor
python -m evals.retrieval_baseline --backend lexical dense hybrid --floor 0.30
```

Same method as ADR 0003: ticket body verbatim as the query, retrieved clauses
rolled up to documents, recall over `relevant_policy_ids`. Two additions:

- **Recall excluding POL-ESC-001.** ADR 0003 classifies ESC-001's approval
  bands as a gating rule decided by arithmetic, not a retrieval target. A
  backend that surfaces ESC-001 more often is not a better retriever for it,
  so backends are compared on recall over the other relevant documents. T-011
  is labeled with ESC-001 alone and drops out, leaving 21 tickets.
- **95% percentile-bootstrap intervals**, using the same seeded resampler as
  `evals/run.py`. Backend differences are paired over tickets.

The second command, rerun against the populated cache, made 0 API calls.

## What the measurement found

At `k=10` (the default) and floor 0.30:

| backend | recall | recall excl. ESC-001 | full recall excl. ESC | ESC-001 | probes empty † | chars |
|---|---|---|---|---|---|---|
| lexical | 0.716 [0.591, 0.833] | 0.810 [0.683, 0.929] | 14/21 | 3/11 | 5/16 | 994 |
| **dense** | 0.727 [0.614, 0.833] | **0.929 [0.857, 0.984]** | **17/21** | 1/11 | **13/16** | 922 |
| hybrid (RRF) | 0.682 [0.553, 0.803] | 0.857 [0.738, 0.952] | 15/21 | 1/11 | 5/16 | 992 |

† Provisional: the probe set is unreviewed. See "Open items".

Paired difference against lexical:

| | recall | recall excl. ESC-001 |
|---|---|---|
| dense | +0.011 [−0.144, +0.155] | **+0.119 [+0.008, +0.246]** |
| hybrid | −0.034 [−0.163, +0.072] | +0.048 [−0.024, +0.127] |

Read plainly:

- **Dense is better on what retrieval is for, by a margin that only just
  clears zero.** The one interval excluding zero has a lower bound of +0.008.
  With 21 tickets that is a measured improvement, not a large one, and the
  Session 8 expansion should either widen or erase the gap.
- **On raw recall the backends tie**, because lexical's lead on ESC-001 (3/11
  against 1/11) offsets dense's lead everywhere else. That lead is incidental
  vocabulary overlap on a class-B document and is not credited.
- **Dense buys the same recall with less context.** Excluding ESC-001, dense
  at k=7 (0.857, 676 chars) already beats lexical at k=10 (0.810, 994 chars).
  Dense's own curve flattens from k=10 to k=12 (0.929 at both), so the default
  limit of 10 stands.
- **Hybrid is not adopted.** ADR 0003 called it the most likely landing spot
  and said it should be chosen on measured recall. Measured, it is below dense
  on every row, and it gives back dense's empty results: a clause sharing one
  word with the query keeps the fused list non-empty, so hybrid's probe
  rejection is lexical's. It stays selectable as an ablation backend.

## ADR 0003's predictions, checked

| ADR 0003 predicted | Measured |
|---|---|
| Fixes class (A), paraphrase and vocabulary | **Held.** Excluding ESC-001, no ticket below 0.50 at k=10. T-012, recall 0.00 under lexical, now retrieves POL-CAN-002; its only miss is ESC-001. |
| No improvement on ESC-001's numeric bands; do not chase | **Held.** 1/11, down from 3/11. Not chased: the band comparison belongs in the gate. |
| Negation may surface both sides of T-016 | **Both sides surface higher:** REF-001.3 rank 1, CAN-002.2 rank 3 (was 6). Whether the agent can tell them apart is for the loop session. |
| The empty result disappears without a floor | **Held, and the baseline was weaker than assumed.** Unfloored dense: 0/16 probes empty. Lexical, assumed to have this property: 5/16. Dense at 0.30: 13/16. † |
| Exact-ID lookup gets worse | **Not measured.** No ticket queries by clause ID. |
| Rankings depend on the model version | Handled by the cache rather than by the name. See Context. |
| Cost and latency land in the per-ticket budget | **Cost on rerun is zero** (0 API calls). First-run token count was not captured. **Latency not measured.** It lands with the loop's p50/p95. |

### The gap tickets cannot be reached by a floor

ADR 0003 said the floor "has to be calibrated against the gap tickets",
implying a good floor would turn them into empty results. **That premise was
wrong.** The gap tickets match the corpus *more* strongly than most covered
tickets do:

```
T-017 (gap)  top-1 similarity 0.545, above 17 of 19 covered tickets
T-018 (gap)  top-1 similarity 0.539, above 16 of 19 covered tickets
T-015 (gap)  top-1 similarity 0.467, above  6 of 19 covered tickets
```

This is the expected shape once it is stated. A within-domain gap, such as
Team → Free with no clause covering that path, is a question *about* the
documents the corpus has. It reads as an ordinary policy question to any
similarity score. A floor separates off-topic text from policy text; it cannot
separate "policy covers this" from "policy stops just short of this".

So the Session 6 guardrail **must not depend on retrieval coming back empty to
detect a gap.** It catches one where it was always going to have to: a
drafted claim with no retrieved passage supporting *that claim*, checked
clause by clause. The floor's job is narrower: off-topic and out-of-scope
text, which it does measurably better than lexical (13/16 against 5/16 †).

## The floor

```
floor   recall    full   probes empty †
 0.25    0.727   9/22         10/16
 0.30    0.727   9/22         13/16    <- chosen by the rule
 0.33    0.705   8/22         15/16    <- variant excluding ESC-001
 0.35    0.636   5/22         15/16
```

**0.30 is fragile, by construction and by sample size.** The rule picks the
edge of the lossless region, and on 22 tickets that edge is one hit:

```
0.3087  T-009  POL-ESC-001 via ESC-001.3   <- pins the chosen floor (class B)
0.3336  T-014  POL-SEA-001 via SEA-001.3   <- would pin the variant
```

The floor is pinned by **an ESC-001 clause, the class-B document this
project says is not a retrieval target.** The rule was written before the
numbers and did not account for that split, so the variant that excludes
ESC-001 is reported alongside it.

**0.30 stands.** Decided 2026-10-14; the label owner delegated the call.

- It is what the rule fixed in advance says. Switching rules after seeing
  where each one lands is the post-hoc choice the rule existed to prevent.
- The trade the variant offers runs the wrong way for this project. 0.33
  rejects two more probes (P-07 at 0.309, P-09 at 0.304), from a 16-probe set
  that is still unreviewed. To get them, it gives up a relevant document on a
  covered ticket (SEA-001 on T-014). The harness's own premise applies: a
  passage the agent did not need is cheap, and a clause it never saw cannot be
  cited.
- Neither value will survive the Session 8 expansion unchanged. A floor
  pinned by one hit out of 22 tickets will move once there are 150.

The floor value depends on ticket recall only. It does **not** depend on the
unreviewed probes.

### ESC-001.6 and the empty result

The probe review surfaced one more clause that retrieval should not be
expected to find: ESC-001.6, *"where policy is silent … escalate rather than
improvising an answer."* It applies to every off-policy query, but it applies
because nothing else matched, not because it resembles the query. The empty
result is what licenses it. So, as ADR 0003 prescribed for the approval bands,
an agent escalating on silence should fetch ESC-001.6 by ID rather than hope
the query reaches it.

## Decision

1. **Dense is the configured backend**: `RETRIEVAL_BACKEND=dense`,
   `RETRIEVAL_DENSE_FLOOR=0.30` in `.env.example`. The comparison that matters
   excludes zero, it needs less context, and it is the only backend that can
   say "nothing relevant" to most out-of-scope text.
2. **Lexical stays the code default.** `RetrievalConfig()` with nothing set is
   lexical, which needs no database, no network and no key. It remains the
   baseline every backend change is measured against, and the backend unit
   tests pin.
3. **A dense or hybrid backend without a floor does not build.** Enforced in
   `RetrievalConfig`, tested.
4. **Hybrid stays as an ablation backend, not adopted.**
5. **The retrieval ablation axis is lexical / dense / hybrid**, run by
   `evals/retrieval_baseline.py`.

## Consequences

- `retrieve_policy` under dense needs Postgres and `OPENAI_API_KEY`. A missing
  one fails at startup, not mid-ticket. A failure mid-run (lost connection,
  provider error after retries) reaches the agent as a `ToolError`, an
  observation it plans around, per CLAUDE.md's failure-handling rules.
- The score in a `retrieve_policy` passage is now cosine similarity for dense
  and an RRF score for hybrid, not TF-IDF. Nothing downstream may compare
  scores across backends.
- Re-embedding is automatic and minimal: changing one clause re-embeds that
  clause only. Changing `EMBEDDING_MODEL` re-embeds everything under the new
  name and leaves the old vectors in place, so switching back is free.

## Open items

- **Human review of `evals/dataset/retrieval_probes.json`**, then set
  `"reviewed": true`. Every figure marked † is provisional until then. The
  file's `review_notes` record the assistant's evidence pass and the one
  interpretive point (ESC-001.6) a reviewer should rule on.
- **Recalibrate the floor** once the Session 8 expansion lands, with the same
  rule and the same command.
- **Clause-level relevance labels** would remove the document rollup that
  flatters every backend, ADR 0003's caveat, still standing.
- **Retrieval latency**, first-run embedding cost, and exact-ID lookup are
  unmeasured.
