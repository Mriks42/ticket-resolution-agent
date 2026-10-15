"""The gate inside the loop: what actually executes, waits or escalates.

`test_gate.py` pins the rules as pure decisions. These pin what the loop does
with them, against the real tools and store: whether money moved, whether a
reply went out, whether a human got the ticket, and what the trace says about
why.
"""

from __future__ import annotations

import json

import pytest

from evals.loader import load_tickets
from evals.schema import Outcome, Ticket
from src.agent.executor import MCPToolExecutor
from src.agent.llm import LLMError
from src.agent.loop import WRITE_SCOPES
from tests.agent_fakes import Misbehaving, ScriptedLLM, completion


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def owner_ticket(tickets) -> Ticket:
    return tickets["T-021"]  # CUST-1834, requested by the account owner


@pytest.fixture
def non_owner_ticket(tickets) -> Ticket:
    return tickets["T-013"]  # CUST-1688, requested by a colleague of the owner


def refund(ticket: Ticket, amount, reason="customer asked for a refund"):
    return completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": amount, "reason": reason})


def reply(ticket: Ticket, body="We have looked into this for you."):
    return completion("send_reply", {"ticket_id": ticket.ticket_id, "body": body})


def gate_events(trace, run_id):
    return [e for e in trace.records(run_id) if e["event"] == "gate"]


# --- bands -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_manager_band_refund_is_held_and_the_ticket_ends_pending(
    make, owner_ticket, ctx, trace
) -> None:
    llm = ScriptedLLM(refund(owner_ticket, 176), reply(owner_ticket, "Submitted for approval (ESC-001.2)."))
    result = await make(llm).run(owner_ticket)

    assert result.outcome is Outcome.PENDING_APPROVAL
    assert result.final_state == "PENDING_APPROVAL"
    seen = json.loads(llm.last_observation(1))
    assert seen["status"] == "pending_approval"
    assert seen["approver"] == "support manager"
    assert seen["policy"]["clause_id"] == "ESC-001.2"
    assert "support manager approval" in seen["policy"]["text"]
    assert ctx.store.records("refunds") == [], "no money moves before approval"
    assert ctx.store.records("replies") == [], "the customer is not told before it is decided"
    assert [(g["tool"], g["verdict"]) for g in gate_events(trace, result.run_id)] == [
        ("issue_refund", "hold"), ("send_reply", "hold"),
    ]


@pytest.mark.asyncio
async def test_the_gates_own_clause_counts_as_retrieved(make, owner_ticket) -> None:
    """The gate hands the model ESC-001.2's text. Citing it back is covered,
    not a hallucination — this is the fetch-by-id ADR 0003 asked for."""
    llm = ScriptedLLM(refund(owner_ticket, 176), reply(owner_ticket, "Per ESC-001.2 a manager will review it."))
    result = await make(llm).run(owner_ticket)
    assert result.final_state == "PENDING_APPROVAL"


@pytest.mark.asyncio
async def test_a_finance_band_refund_names_finance(make, owner_ticket) -> None:
    llm = ScriptedLLM(refund(owner_ticket, 1584), reply(owner_ticket))
    result = await make(llm).run(owner_ticket)
    seen = json.loads(llm.last_observation(1))
    assert seen["approver"] == "finance"
    assert seen["policy"]["clause_id"] == "ESC-001.3"
    assert result.final_state == "PENDING_APPROVAL"


@pytest.mark.asyncio
async def test_refunds_are_banded_on_the_ticket_total(make, owner_ticket, ctx, trace) -> None:
    """$30 is self-serve; a further $25 on the same ticket is not."""
    llm = ScriptedLLM(refund(owner_ticket, 30), refund(owner_ticket, 25), reply(owner_ticket))
    result = await make(llm).run(owner_ticket)

    assert result.final_state == "PENDING_APPROVAL"
    assert [r["amount"] for r in ctx.store.records("refunds")] == [30.0]
    assert json.loads(llm.last_observation(2))["approver"] == "support manager"
    decisions = [e for e in gate_events(trace, result.run_id) if e["tool"] == "issue_refund"]
    assert [d["facts"]["refunded_before"] for d in decisions] == [0, 30]


@pytest.mark.asyncio
async def test_writes_go_through_the_write_server_only(make, owner_ticket, server) -> None:
    llm = ScriptedLLM(refund(owner_ticket, 20), reply(owner_ticket))
    result = await make(llm).run(owner_ticket)

    assert result.final_state == "RESOLVE"
    writes = [r for r in server.audit.records() if r["tool"] in ("issue_refund", "send_reply")]
    assert [r["status"] for r in writes] == ["ok", "ok"]
    assert all(set(r["caller_scopes"]) == {s.value for s in WRITE_SCOPES} for r in writes)


@pytest.mark.asyncio
async def test_an_unusable_amount_is_reported_back_to_the_model(make, owner_ticket, ctx) -> None:
    llm = ScriptedLLM(refund(owner_ticket, "lots"), reply(owner_ticket))
    result = await make(llm).run(owner_ticket)
    assert json.loads(llm.last_observation(1))["error"] == "invalid_arguments"
    assert ctx.store.records("refunds") == []
    assert result.final_state == "RESOLVE"


# --- the owner check -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refund_for_a_non_owner_escalates_and_is_filed(
    make, non_owner_ticket, ctx, server
) -> None:
    llm = ScriptedLLM(refund(non_owner_ticket, 88))
    result = await make(llm).run(non_owner_ticket)

    assert result.final_state == "ESCALATE"
    assert "ESC-001.4" in result.stop_reason
    assert ctx.store.records("refunds") == []
    [filed] = ctx.store.records("escalations")
    assert "ESC-001.4" in filed["reason"]
    lookups = [r for r in server.audit.records() if r["tool"] == "lookup_customer"]
    assert [r["run_id"] for r in lookups] == [result.run_id], (
        "the gate verifies ownership itself, through an audited read"
    )


@pytest.mark.asyncio
async def test_a_non_owner_can_still_be_answered(make, non_owner_ticket, ctx) -> None:
    """Money is blocked for non-owners; answers are not (decided with the
    label owner, so Session 8 can label non-owner tickets other than escalate)."""
    llm = ScriptedLLM(reply(non_owner_ticket))
    result = await make(llm).run(non_owner_ticket)
    assert result.final_state == "RESOLVE"
    assert len(ctx.store.records("replies")) == 1


# --- coverage, enforced by the guardrail ahead of the gate --------------------------------


@pytest.mark.asyncio
async def test_a_reply_citing_an_unretrieved_clause_is_a_hallucinated_policy_event(
    make, owner_ticket, ctx, trace
) -> None:
    """Coverage is the guardrail's (ADR 0007), and it runs before the gate."""
    llm = ScriptedLLM(reply(owner_ticket, "Annual renewals are refundable within 14 days (REF-001.3)."))
    result = await make(llm).run(owner_ticket)

    assert result.final_state == "ESCALATE"
    assert "REF-001.3" in result.stop_reason
    assert ctx.store.records("replies") == []
    assert gate_events(trace, result.run_id) == [], "blocked before the gate decided"
    [check] = [e for e in trace.records(result.run_id) if e["event"] == "guardrail"]
    assert check["violations"][0]["kind"] == "citation_not_retrieved"
    assert len(ctx.store.records("escalations")) == 1


@pytest.mark.asyncio
async def test_a_reply_citing_a_retrieved_clause_is_sent(make, owner_ticket, ctx) -> None:
    llm = ScriptedLLM(
        completion("retrieve_policy", {"query": "annual renewal charge refundable"}),
        reply(owner_ticket, "Annual renewals are refundable within 14 days (REF-001.3)."),
    )
    result = await make(llm).run(owner_ticket)
    assert result.final_state == "RESOLVE"
    assert len(ctx.store.records("replies")) == 1


# --- escalation -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_wins_over_a_held_refund(make, owner_ticket, ctx) -> None:
    """T-021's shape: a manager-band refund, then the model hands off. The gate
    does not override the hand-off; the human gets the held refund with it."""
    llm = ScriptedLLM(
        refund(owner_ticket, 176),
        completion("escalate", {"ticket_id": owner_ticket.ticket_id, "reason": "customer also wants closure"}),
    )
    result = await make(llm).run(owner_ticket)

    assert result.final_state == "ESCALATE"
    assert ctx.store.records("refunds") == []
    [filed] = ctx.store.records("escalations")
    held = [item["held_for_approval"] for item in filed["trace"] if "held_for_approval" in item]
    assert held and held[0][0]["args"]["amount"] == 176
    assert held[0][0]["approver"] == "support manager"


@pytest.mark.asyncio
async def test_exactly_one_escalation_is_filed_when_the_model_escalates(make, owner_ticket, ctx) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": owner_ticket.ticket_id, "reason": "r"}))
    await make(llm).run(owner_ticket)
    assert len(ctx.store.records("escalations")) == 1


@pytest.mark.asyncio
async def test_a_stop_by_the_loop_still_files_an_escalation(make, owner_ticket, ctx, trace) -> None:
    """A budget stop that queued nothing would lose the ticket."""
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": owner_ticket.customer_id}),
        completion("retrieve_policy", {"query": "refund"}),
    )
    result = await make(llm, max_steps=1).run(owner_ticket)

    assert result.final_state == "ESCALATE"
    [filed] = ctx.store.records("escalations")
    assert "step budget" in filed["reason"]
    [event] = [e for e in trace.records(result.run_id) if e["event"] == "escalation_filed"]
    assert event["filed_by"] == "system"
    assert event["status"] == "ok"


@pytest.mark.asyncio
async def test_a_failed_run_still_reaches_a_human(make, owner_ticket, ctx) -> None:
    result = await make(ScriptedLLM(LLMError("chat request rejected, HTTP 401"))).run(owner_ticket)
    assert result.final_state == "FAILED"
    [filed] = ctx.store.records("escalations")
    assert "401" in filed["reason"]


@pytest.mark.asyncio
async def test_a_reply_that_cannot_be_sent_escalates(make, owner_ticket, ctx) -> None:
    llm = ScriptedLLM(reply(owner_ticket))
    broken = MCPToolExecutor(Misbehaving(text="<<not json>>"), timeout=5)
    result = await make(llm, write_executor=broken).run(owner_ticket)
    assert result.final_state == "ESCALATE"
    assert "could not be sent" in result.stop_reason


@pytest.mark.asyncio
async def test_a_refund_that_times_out_escalates_as_unknown(make, owner_ticket) -> None:
    """Whether the money moved is unknown, so a human must check — the run
    must not carry on as if the refund had simply failed."""
    llm = ScriptedLLM(refund(owner_ticket, 20), reply(owner_ticket))
    slow = MCPToolExecutor.for_writes(Misbehaving(delay=0.3), timeout=0.05)
    result = await make(llm, write_executor=slow).run(owner_ticket)
    assert result.final_state == "ESCALATE"
    assert "unknown" in result.stop_reason


def test_the_agent_wires_a_non_retrying_writer(monkeypatch, tmp_path) -> None:
    from src.agent.loop import LoopAgent

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("TRA_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("RETRIEVAL_BACKEND", "lexical")
    agent = LoopAgent()
    assert agent._writer._retry.max_attempts == 1
    assert agent._executor._retry.max_attempts > 1
