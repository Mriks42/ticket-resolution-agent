"""The guardrail inside the loop: blocking, detect-only, and what counts as
the customer's own data.

`test_guardrail.py` pins the check itself. These pin what the loop feeds it
(retrieved passages, clauses the gate showed, the ticket, the account, refund
amounts) and what happens after: in block mode a flagged reply is not sent
and the ticket escalates; in detect mode the reply goes out and the violation
is still counted. That pair is the "with vs without guardrail" ablation.
"""

from __future__ import annotations

import pytest

from evals.loader import load_tickets
from evals.schema import Outcome, Ticket
from src.agent.config import AgentConfig
from tests.agent_fakes import ScriptedLLM, completion


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def ticket(tickets) -> Ticket:
    return tickets["T-021"]  # owner request; body mentions "$176" and "eight seats"


def reply(ticket: Ticket, body: str):
    return completion("send_reply", {"ticket_id": ticket.ticket_id, "body": body})


def guardrail_events(trace, run_id):
    return [e for e in trace.records(run_id) if e["event"] == "guardrail"]


INVENTED_FEE = "Refunds carry a $15 processing fee."


# --- block vs detect ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_mode_stops_a_hallucinated_reply(make, ticket, ctx, trace) -> None:
    result = await make(ScriptedLLM(reply(ticket, INVENTED_FEE))).run(ticket)

    assert result.final_state == "ESCALATE"
    assert "hallucinated policy" in result.stop_reason
    assert ctx.store.records("replies") == []
    assert len(ctx.store.records("escalations")) == 1
    [event] = guardrail_events(trace, result.run_id)
    assert event["mode"] == "block"
    assert [v["kind"] for v in event["violations"]] == ["unsupported_fact"]
    assert result.policy_violations and result.policy_violations[0].startswith("unsupported_fact")


@pytest.mark.asyncio
async def test_detect_mode_sends_it_and_still_counts_it(make, ticket, ctx, trace) -> None:
    result = await make(ScriptedLLM(reply(ticket, INVENTED_FEE)), guardrail_mode="detect").run(ticket)

    assert result.final_state == "RESOLVE"
    assert len(ctx.store.records("replies")) == 1
    assert result.policy_violations, "the off arm of the ablation must still measure"
    assert guardrail_events(trace, result.run_id)[0]["mode"] == "detect"


@pytest.mark.asyncio
async def test_a_clean_reply_has_no_violations(make, ticket) -> None:
    result = await make(ScriptedLLM(reply(ticket, "We are looking into this for you."))).run(ticket)
    assert result.final_state == "RESOLVE"
    assert result.policy_violations == []


# --- what the loop feeds the check -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_retrieved_clause_grounds_its_figure(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("retrieve_policy", {"query": "annual renewal charge refundable"}),
        reply(ticket, "Annual renewals are refundable within 14 days (REF-001.3)."),
    )
    assert (await make(llm).run(ticket)).final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_a_figure_from_the_wrong_retrieved_clause_is_miscited(make, ticket, trace) -> None:
    llm = ScriptedLLM(
        completion("retrieve_policy", {"query": "annual renewal charge refundable"}),
        reply(ticket, "Annual renewals are refundable within 30 days (REF-001.3)."),
    )
    result = await make(llm).run(ticket)
    assert result.final_state == "ESCALATE"
    assert guardrail_events(trace, result.run_id)[0]["violations"][0]["kind"] == "miscited_fact"


@pytest.mark.asyncio
async def test_the_customers_own_figures_are_grounded(make, ticket) -> None:
    """$176 and eight seats are in the ticket body; they are the customer's
    facts, not policy claims."""
    llm = ScriptedLLM(reply(ticket, "I can see the $176 charge for your 8 seats."))
    assert (await make(llm).run(ticket)).final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_account_figures_from_a_lookup_are_grounded(make, tickets) -> None:
    t = tickets["T-001"]  # body mentions no amount; the account says $29
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": t.customer_id}),
        reply(t, "Your last charge was $29."),
    )
    assert (await make(llm).run(t)).final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_a_proposed_refund_amount_and_the_gates_clause_are_grounded(make, tickets) -> None:
    t = tickets["T-017"]  # $290 annual charge, owner request
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": t.ticket_id, "amount": 290, "reason": "within the window"}),
        reply(t, "Your $290 refund needs approval for amounts between USD 50.01 and USD 500 (ESC-001.2)."),
    )
    result = await make(llm).run(t)
    assert result.final_state == "PENDING_APPROVAL"
    assert result.policy_violations == []


@pytest.mark.asyncio
async def test_a_refund_reason_is_checked_too(make, ticket, ctx) -> None:
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": 20, "reason": "inside the 45-day window"}),
    )
    result = await make(llm).run(ticket)
    assert result.final_state == "ESCALATE"
    assert ctx.store.records("refunds") == []


@pytest.mark.asyncio
async def test_the_guardrail_runs_before_the_gate(make, ticket) -> None:
    """A hallucinated reply is never parked for approval behind a held refund."""
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": 176, "reason": "pilot ended"}),
        reply(ticket, INVENTED_FEE),
    )
    result = await make(llm).run(ticket)
    assert result.final_state == "ESCALATE"
    assert result.outcome is Outcome.ESCALATE


# --- config ---------------------------------------------------------------------------------


def test_guardrail_mode_defaults_to_block_and_rejects_nonsense() -> None:
    assert AgentConfig().guardrail_mode == "block"
    assert AgentConfig.from_env({"GUARDRAIL_MODE": "detect"}).guardrail_mode == "detect"
    with pytest.raises(ValueError, match="guardrail_mode"):
        AgentConfig(guardrail_mode="off")
