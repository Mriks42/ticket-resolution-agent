"""The HTTP API, driven through ASGI with the scripted LLM.

FastAPI's own test client needs httpx, which is not a project dependency, so
`asgi()` speaks the ASGI protocol directly: the request goes through the real
app — routing, validation, the key checks — exactly as uvicorn would deliver
it, without a socket. Tools, gate, store, audit log and trace are the
agent-loop rig's, so what an approval executes is checked where it lands.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from evals.loader import load_tickets
from evals.schema import Ticket
from src.agent.pricing import Price, SpendGuard
from src.api.app import create_app
from src.api.service import AgentService
from tests.agent_fakes import ScriptedLLM, completion

SUBMIT, APPROVE = "submit-key-for-tests", "approve-key-for-tests"
TODAY = date(2026, 8, 24)


async def asgi(app, method: str, path: str, *, body=None, key: str | None = None):
    """One request through the ASGI interface. Returns (status, parsed JSON)."""
    payload = json.dumps(body).encode() if body is not None else b""
    headers = [(b"content-type", b"application/json")]
    if key is not None:
        headers.append((b"x-api-key", key.encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "headers": headers,
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, (json.loads(raw) if raw else None)


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def owner_ticket(tickets) -> Ticket:
    return tickets["T-021"]  # CUST-1834, the owner asking


def submitted(ticket: Ticket) -> dict:
    """What a caller sends: the four fields, no labels."""
    return {k: getattr(ticket, k) for k in ("ticket_id", "customer_id", "requester_email", "body")}


@pytest.fixture
def api(make, ctx):
    """Build (app, service) around a scripted LLM."""
    def build(llm, **kw):
        service = AgentService(agent=make(llm, **kw), context=ctx, today=lambda: TODAY)
        return create_app(service, submit_key=SUBMIT, approve_key=APPROVE), service
    return build


def escalate(ticket: Ticket):
    return completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"})


def held_refund(ticket: Ticket, amount=176):
    return [
        completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": amount, "reason": "in window"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "Your refund is with a manager."}),
    ]


# --- keys --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_needs_no_key(api, owner_ticket) -> None:
    app, _ = api(ScriptedLLM())
    assert await asgi(app, "GET", "/health") == (200, {"status": "ok"})


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, "wrong", APPROVE])
async def test_submitting_needs_the_submit_key(api, owner_ticket, key) -> None:
    app, _ = api(ScriptedLLM())
    status, _ = await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=key)
    assert status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, SUBMIT])
async def test_approvals_need_the_approve_key(api, key) -> None:
    app, _ = api(ScriptedLLM())
    assert (await asgi(app, "GET", "/approvals", key=key))[0] == 401
    body = {"decision": "approve", "decided_by": "x"}
    assert (await asgi(app, "POST", "/approvals/apr-x", body=body, key=key))[0] == 401


@pytest.mark.parametrize(("submit", "approve"), [("", "a"), ("s", ""), ("same", "same")])
def test_the_app_refuses_to_build_without_two_distinct_keys(ctx, make, submit, approve) -> None:
    service = AgentService(agent=make(ScriptedLLM()), context=ctx)
    with pytest.raises(ValueError):
        create_app(service, submit_key=submit, approve_key=approve)


# --- tickets and runs ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_ticket_is_queued_then_run_in_the_background(api, owner_ticket) -> None:
    app, service = api(ScriptedLLM(escalate(owner_ticket)))
    status, body = await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=SUBMIT)
    assert status == 202
    assert body["status"] == "queued"

    await service.idle()
    status, run = await asgi(app, "GET", f"/runs/{body['run_id']}", key=SUBMIT)
    assert status == 200
    assert run["status"] == "finished"
    assert run["result"]["outcome"] == "escalate"
    assert run["events"][0]["event"] == "run_started"
    assert run["events"][0]["ticket_id"] == owner_ticket.ticket_id


@pytest.mark.asyncio
async def test_runs_happen_one_at_a_time_in_submission_order(api, owner_ticket, trace) -> None:
    app, service = api(ScriptedLLM(escalate(owner_ticket), escalate(owner_ticket)))
    first = (await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=SUBMIT))[1]["run_id"]
    second = (await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=SUBMIT))[1]["run_id"]
    await service.idle()

    lines = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    order = [(e["run_id"], e["event"]) for e in lines if e["event"] in ("run_started", "run_finished")]
    assert order == [
        (first, "run_started"), (first, "run_finished"), (second, "run_started"), (second, "run_finished"),
    ]


@pytest.mark.asyncio
async def test_an_unknown_customer_is_refused_before_anything_runs(api, owner_ticket) -> None:
    app, service = api(ScriptedLLM())
    status, body = await asgi(
        app, "POST", "/tickets", body={**submitted(owner_ticket), "customer_id": "CUST-9999"}, key=SUBMIT,
    )
    assert status == 404
    assert "CUST-9999" in body["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"body": ""}, {"labels": {"outcome": "resolve"}}])
async def test_a_malformed_ticket_is_rejected(api, owner_ticket, change) -> None:
    """Including one that tries to carry labels: nothing beyond the four fields."""
    app, _ = api(ScriptedLLM())
    status, _ = await asgi(app, "POST", "/tickets", body={**submitted(owner_ticket), **change}, key=SUBMIT)
    assert status == 422


@pytest.mark.asyncio
async def test_an_unknown_run_is_404(api) -> None:
    app, _ = api(ScriptedLLM())
    assert (await asgi(app, "GET", "/runs/run-nope", key=SUBMIT))[0] == 404


@pytest.mark.asyncio
async def test_a_run_that_raises_is_recorded_as_failed(api, owner_ticket) -> None:
    """The spend cap stops a run loudly; the service records it and stays up."""
    guard = SpendGuard(cap_usd=0.00001, price=Price(input_per_mtok=0.15, output_per_mtok=0.60))
    app, service = api(ScriptedLLM(escalate(owner_ticket)), spend=guard)
    run_id = (await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=SUBMIT))[1]["run_id"]
    await service.idle()
    _, run = await asgi(app, "GET", f"/runs/{run_id}", key=SUBMIT)
    assert run["status"] == "failed"
    assert "SpendCapExceeded" in run["error"]


@pytest.mark.asyncio
async def test_a_finished_run_is_still_found_after_a_restart(api, make, ctx, owner_ticket) -> None:
    app, service = api(ScriptedLLM(escalate(owner_ticket)))
    run_id = (await asgi(app, "POST", "/tickets", body=submitted(owner_ticket), key=SUBMIT))[1]["run_id"]
    await service.idle()

    restarted = create_app(
        AgentService(agent=make(ScriptedLLM()), context=ctx), submit_key=SUBMIT, approve_key=APPROVE,
    )
    status, run = await asgi(restarted, "GET", f"/runs/{run_id}", key=SUBMIT)
    assert (status, run["status"], run["result"]["outcome"]) == (200, "finished", "escalate")


# --- approvals ---------------------------------------------------------------------------------


async def pending_run(api, ticket):
    app, service = api(ScriptedLLM(*held_refund(ticket)))
    run_id = (await asgi(app, "POST", "/tickets", body=submitted(ticket), key=SUBMIT))[1]["run_id"]
    await service.idle()
    return app, run_id


@pytest.mark.asyncio
async def test_a_pending_run_opens_an_approval_with_what_was_held(api, owner_ticket, ctx) -> None:
    app, run_id = await pending_run(api, owner_ticket)
    _, run = await asgi(app, "GET", f"/runs/{run_id}", key=SUBMIT)
    assert run["result"]["outcome"] == "pending_approval"

    status, body = await asgi(app, "GET", "/approvals", key=APPROVE)
    [approval] = body["pending"]
    assert approval["approval_id"] == run["result"]["approval_id"]
    assert approval["run_id"] == run_id
    assert [h["tool"] for h in approval["held"]] == ["issue_refund", "send_reply"]
    assert ctx.store.records("refunds") == [] and ctx.store.records("replies") == []


@pytest.mark.asyncio
async def test_approving_executes_the_refund_then_the_reply(api, owner_ticket, ctx, server) -> None:
    app, run_id = await pending_run(api, owner_ticket)
    approval_id = (await asgi(app, "GET", "/approvals", key=APPROVE))[1]["pending"][0]["approval_id"]

    status, decided = await asgi(
        app, "POST", f"/approvals/{approval_id}",
        body={"decision": "approve", "decided_by": "Sam, support manager"}, key=APPROVE,
    )
    assert (status, decided["status"]) == (200, "approved")
    assert [e["tool"] for e in decided["executed"]] == ["issue_refund", "send_reply"]
    assert [r["amount"] for r in ctx.store.records("refunds")] == [176.0]
    assert [r["body"] for r in ctx.store.records("replies")] == ["Your refund is with a manager."]
    assert ctx.billing.refunds(owner_ticket.customer_id) == [
        {"date": "2026-08-24", "amount_usd": 176.0, "ticket_id": owner_ticket.ticket_id},
    ]
    assert (await asgi(app, "GET", "/approvals", key=APPROVE))[1]["pending"] == []

    events = (await asgi(app, "GET", f"/runs/{run_id}", key=SUBMIT))[1]["events"]
    assert events[-1]["event"] == "approval_decided"
    assert events[-1]["decided_by"] == "Sam, support manager"


@pytest.mark.asyncio
async def test_rejecting_executes_nothing(api, owner_ticket, ctx) -> None:
    app, _ = await pending_run(api, owner_ticket)
    approval_id = (await asgi(app, "GET", "/approvals", key=APPROVE))[1]["pending"][0]["approval_id"]
    _, decided = await asgi(
        app, "POST", f"/approvals/{approval_id}",
        body={"decision": "reject", "decided_by": "Sam", "note": "outside policy"}, key=APPROVE,
    )
    assert (decided["status"], decided["executed"]) == ("rejected", [])
    assert ctx.store.records("refunds") == [] and ctx.store.records("replies") == []


@pytest.mark.asyncio
async def test_an_approval_is_decided_once(api, owner_ticket) -> None:
    app, _ = await pending_run(api, owner_ticket)
    approval_id = (await asgi(app, "GET", "/approvals", key=APPROVE))[1]["pending"][0]["approval_id"]
    body = {"decision": "approve", "decided_by": "Sam"}
    assert (await asgi(app, "POST", f"/approvals/{approval_id}", body=body, key=APPROVE))[0] == 200
    assert (await asgi(app, "POST", f"/approvals/{approval_id}", body=body, key=APPROVE))[0] == 409


@pytest.mark.asyncio
async def test_an_unknown_approval_is_404(api) -> None:
    app, _ = api(ScriptedLLM())
    body = {"decision": "approve", "decided_by": "Sam"}
    assert (await asgi(app, "POST", "/approvals/apr-nope", body=body, key=APPROVE))[0] == 404
