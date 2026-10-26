# 12. The API: background runs, approvals, two keys

## Status

Accepted. Built and tested through ASGI, with a smoke test against the real
agent under uvicorn. Not deployed; Terraform is next.

## Context

CLAUDE.md puts a FastAPI service layer in front of the agent. Nothing
completed the HITL loop yet: a refund the gate held for approval (ADR 0006)
stayed held forever, with no way for a human to release or refuse it.

Three choices were made with the project owner before the code:

| question | choice | rejected |
|---|---|---|
| scope | tickets and approvals | tickets only, which leaves held refunds held |
| execution | a background job: 202 now, poll for the result | synchronous: a run may take the full 120 s wall-clock budget, and API Gateway in front of Lambda cuts requests off at 29 s |
| auth | two API keys, one to submit and one to approve | one key, with which whoever submits a ticket could release its refund; no auth |

## Decisions

**Routes.** `POST /tickets` queues a run and returns its id with 202.
`GET /runs/{id}` returns the run's status, result and trace events.
`GET /approvals` lists what is waiting. `POST /approvals/{id}` approves or
rejects. `GET /health` is open. The ticket body is the four fields the agent
is shown, and an unknown field, labels included, is a 422.

**Runs execute one at a time, in submission order**, behind a single lock.
That keeps per-run embedding cost exact (ADR 0009), and keeps memory and
billing writes in the order an eval run makes them. One instance therefore
handles one ticket at a time, so throughput scales by instances. A run that
raises, such as one that hits the spend cap, is recorded as failed with its
error, and the service stays up.

**The trace is the record of a run.** Status is held in memory while the
process lives, and events are always read from the trace, so a finished run
can be looked up after a restart. A run queued but not started when the
process stops is lost.

**Approvals are an append-only log** (`var/approvals/approvals.jsonl`), in
the trace's format (ADR 0001). A `held` event is written when a run ends in
PENDING_APPROVAL, carrying what the gate held. A `decided` event is written
when a human approves or rejects. State is the fold of the events. A run that
ends in ESCALATE with something held does not open an approval: escalation
wins over anything held (ADR 0006), and the escalation already carries it.

**Approving executes what was held, through the write-scoped server**:
refunds first, then the held reply, each audited under the original run id,
with no retries (ADR 0006). Executed refunds join the account's billing
history (ADR 0011). The first action that does not succeed stops the rest, so
a customer is never told about a refund that did not go through. The
decision, who made it, and what each action returned go into the approvals
log and into the run's trace as `approval_decided`. Rejecting executes
nothing. An approval is decided once; a second decision is a 409.

**Two keys, one per side of the gate**, from `TRA_API_KEY_SUBMIT` and
`TRA_API_KEY_APPROVE`, sent as `X-API-Key` and compared in constant time.
The app refuses to build without both, or with the same key twice.

**The agent is the eval agent.** `create_app_from_env` builds the same
`LoopAgent` as `evals/run.py`, from the same environment, so the API serves
the configuration the numbers describe.

**Tests drive the app through ASGI directly.** FastAPI's test client needs
httpx, which is not a dependency, and CLAUDE.md asks before adding one. A
small helper sends real ASGI requests through the real app, so routing,
validation and the key checks are all exercised.

## What it costs

- **Approvers are not distinguished.** The gate names an approver for each
  held refund: a support manager, or finance above USD 500. Anyone with the
  approve key can release either; the API records who they said they were
  but cannot check it. Per-approver keys or real identity belong with the
  deployment.
- **Memory is not updated on approval.** The run's close recorded the refund
  as held, which was true when it was written. Billing history, the
  authority on refunds, is updated.
- **Single process.** Runs, the lock and the in-memory status assume one
  instance per memory namespace and billing history. Several instances would
  each have their own memory, so a customer's tickets would need routing to
  one instance, or memory and billing would need moving fully into Postgres.
- **Ticket ids are not deduplicated.** Submitting the same ticket twice runs
  it twice.
