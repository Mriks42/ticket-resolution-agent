"""Memory on and off inside the loop.

Off is the stateless baseline of the memory ablation, so it has to be
stateless in fact, not just missing a tool: the history search is not offered,
a call to it is not run, and `lookup_customer` reaches neither the model nor
the guardrail with the customer's past tickets in it. On, the loop behaves as
it did before the switch existed. ADR 0008.
"""

from __future__ import annotations

import json

import pytest

from evals.loader import load_tickets
from evals.schema import Ticket
from src.agent.config import AgentConfig
from src.tools.catalog import CATALOG
from tests.agent_fakes import ScriptedLLM, completion


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def ticket(tickets) -> Ticket:
    return tickets["T-020"]  # CUST-1520: two prior contacts, both slow responses


def escalate(ticket: Ticket):
    return completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"})


@pytest.mark.asyncio
async def test_memory_off_does_not_offer_the_history_search(make, ticket) -> None:
    llm = ScriptedLLM(escalate(ticket))
    await make(llm, memory=False).run(ticket)
    assert llm.tools_offered == [name for name in CATALOG if name != "search_past_tickets"]


@pytest.mark.asyncio
async def test_memory_off_does_not_run_a_history_search(make, ticket, server) -> None:
    llm = ScriptedLLM(
        completion("search_past_tickets", {"query": "slow response"}),
        escalate(ticket),
    )
    result = await make(llm, memory=False).run(ticket)

    assert json.loads(llm.last_observation(1))["error"] == "unknown_tool"
    assert [r["tool"] for r in server.audit.records() if r["tool"] == "search_past_tickets"] == []
    assert result.tool_calls[0].tool.value == "escalate"


@pytest.mark.asyncio
async def test_memory_off_strips_history_from_the_account(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        escalate(ticket),
    )
    await make(llm, memory=False).run(ticket)

    account = json.loads(llm.last_observation(1))
    assert "past_tickets" not in account
    assert account["customer_id"] == ticket.customer_id
    assert account["seats"] == 6


@pytest.mark.asyncio
async def test_memory_on_shows_the_account_history(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        escalate(ticket),
    )
    await make(llm).run(ticket)
    assert [p["ticket_id"] for p in json.loads(llm.last_observation(1))["past_tickets"]] == [
        "H-4471", "H-5088",
    ]


@pytest.mark.parametrize(("memory", "final_state"), [(True, "RESOLVE"), (False, "ESCALATE")])
@pytest.mark.asyncio
async def test_history_grounds_a_reply_only_with_memory_on(make, ticket, memory, final_state) -> None:
    """'51 hours' is in H-4471's resolution and nowhere in the ticket or the
    account figures. With memory off the guardrail must not have seen it."""
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("send_reply", {
            "ticket_id": ticket.ticket_id,
            "body": "I can see your first response in June took 51 hours.",
        }),
    )
    assert (await make(llm, memory=memory).run(ticket)).final_state == final_state


@pytest.mark.asyncio
async def test_run_started_records_the_memory_setting(make, ticket, trace) -> None:
    result = await make(ScriptedLLM(escalate(ticket)), memory=False).run(ticket)
    [started] = [e for e in trace.records(result.run_id) if e["event"] == "run_started"]
    assert started["memory"] is False


def test_memory_defaults_to_on_and_rejects_nonsense() -> None:
    assert AgentConfig().memory is True
    assert AgentConfig.from_env({"MEMORY_ENABLED": "false"}).memory is False
    assert AgentConfig.from_env({"MEMORY_ENABLED": "TRUE"}).memory is True
    with pytest.raises(ValueError, match="MEMORY_ENABLED"):
        AgentConfig.from_env({"MEMORY_ENABLED": "off"})
    with pytest.raises(ValueError, match="memory"):
        AgentConfig(memory="false")
