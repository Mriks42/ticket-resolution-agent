"""Memory on and off inside the loop.

Off is the stateless baseline of the memory ablation. It has to be stateless
in fact — no history reaches the model or the guardrail, and nothing is
written — and it has to change nothing else. The first version hid the
history tool and the `past_tickets` field, and that interface change alone
flipped a ticket with no history three runs out of three. So off keeps the
same tools and the same fields and empties them: for a customer with no
history, the two arms see byte-identical input. On, every run is remembered
at close, as the customer's words and a fixed outcome, never the model's
prose. ADR 0008.
"""

from __future__ import annotations

import json

import pytest

from evals.loader import load_tickets
from evals.schema import Ticket
from src.agent.config import AgentConfig
from src.agent.guardrail import account_grounding
from src.agent.llm import LLMError
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
async def test_memory_off_offers_the_same_tools(make, ticket) -> None:
    llm = ScriptedLLM(escalate(ticket))
    await make(llm, memory=False).run(ticket)
    assert llm.tools_offered == list(CATALOG)


@pytest.mark.asyncio
async def test_memory_off_history_search_finds_nothing(make, ticket, server) -> None:
    llm = ScriptedLLM(
        completion("search_past_tickets", {"query": "slow response"}),
        escalate(ticket),
    )
    await make(llm, memory=False).run(ticket)

    assert json.loads(llm.last_observation(1)) == {
        "query": "slow response", "customer_id": ticket.customer_id, "matches": [],
    }
    assert [r["tool"] for r in server.audit.records()][0] == "search_past_tickets"


@pytest.mark.asyncio
async def test_memory_off_shows_an_empty_history_in_the_same_place(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        escalate(ticket),
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        escalate(ticket),
    )
    await make(llm, memory=False).run(ticket)
    await make(llm).run(ticket)

    off, on = json.loads(llm.last_observation(1)), json.loads(llm.last_observation(3))
    assert off["past_tickets"] == []
    assert on["past_tickets"]
    assert list(off) == list(on)[: len(off)]
    assert {k: v for k, v in off.items() if k != "past_tickets"} == {
        k: v for k, v in on.items() if k not in ("past_tickets", "memory")
    }


@pytest.mark.asyncio
async def test_without_history_both_arms_see_identical_input(make, tickets) -> None:
    """The point of the ablation: where there is no history, the switch must
    change nothing the model sees."""
    t = tickets["T-022"]  # CUST-1867: no history; flipped 3/3 under the first off arm
    script = lambda: [  # noqa: E731 - one script, run twice
        completion("lookup_customer", {"customer_id": t.customer_id}),
        completion("search_past_tickets", {"query": "renewal charge"}),
        escalate(t),
    ]
    on, off = ScriptedLLM(*script()), ScriptedLLM(*script())
    await make(on).run(t)
    await make(off, memory=False).run(t)

    assert on.tools_offered == off.tools_offered
    assert [on.last_observation(i) for i in (1, 2)] == [off.last_observation(i) for i in (1, 2)]


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


# --- what a closed run leaves behind ------------------------------------------------


def history(ctx, customer_id: str):
    return ctx.memory.episodes.history(customer_id)


@pytest.mark.asyncio
async def test_a_closed_ticket_is_remembered(make, tickets, ctx, trace) -> None:
    t = tickets["T-001"]
    llm = ScriptedLLM(
        completion("send_reply", {"ticket_id": t.ticket_id, "body": "We are looking into this for you."}),
    )
    result = await make(llm).run(t)

    [episode] = history(ctx, t.customer_id)
    assert (episode.ticket_id, episode.summary) == (t.ticket_id, t.body)
    assert episode.resolution == "Resolved by the agent; a reply was sent."
    [written] = [e for e in trace.records(result.run_id) if e["event"] == "memory_written"]
    assert written["episode"]["ticket_id"] == t.ticket_id
    assert written["facts"] == ctx.memory.facts.get(t.customer_id)
    assert written["namespace"] == ctx.memory.namespace


@pytest.mark.asyncio
async def test_memory_off_writes_nothing(make, tickets, ctx, trace) -> None:
    t = tickets["T-001"]
    result = await make(ScriptedLLM(escalate(t)), memory=False).run(t)
    assert history(ctx, t.customer_id) == []
    assert [e for e in trace.records(result.run_id) if e["event"] == "memory_written"] == []


@pytest.mark.asyncio
async def test_a_repeat_customer_sees_the_earlier_ticket(make, tickets) -> None:
    """What memory is for, and what the held-out set cannot test: no customer
    there writes in twice."""
    first = tickets["T-001"]
    second = first.model_copy(update={"ticket_id": "T-001-B", "body": "Following up on my last message."})
    llm = ScriptedLLM(
        escalate(first),
        completion("lookup_customer", {"customer_id": second.customer_id}),
        escalate(second),
    )
    loop = make(llm)
    await loop.run(first)
    await loop.run(second)

    account = json.loads(llm.last_observation(2))
    assert [(p["ticket_id"], p["summary"], p["resolution"]) for p in account["past_tickets"]] == [
        ("T-001", first.body, "Escalated to a human."),
    ]
    assert account["memory"]["tickets_closed"] == 1
    assert [e["ticket_id"] for e in account["memory"]["escalations"]] == ["T-001"]


@pytest.mark.asyncio
async def test_memory_off_hides_what_an_earlier_run_remembered(make, tickets) -> None:
    first = tickets["T-001"]
    second = first.model_copy(update={"ticket_id": "T-001-B", "body": "Following up on my last message."})
    await make(ScriptedLLM(escalate(first))).run(first)
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": second.customer_id}),
        escalate(second),
    )
    await make(llm, memory=False).run(second)

    account = json.loads(llm.last_observation(1))
    assert account["past_tickets"] == []
    assert "memory" not in account


def test_remembered_facts_are_not_grounding() -> None:
    """Refund amounts reach a later reply's grounding through the episode
    text in `past_tickets`, as the customer's history; the facts block adds
    nothing the guardrail reads."""
    texts, facts = account_grounding({"memory": {"refunds": [{"amount": 77.0, "status": "issued"}]}})
    assert (texts, facts) == ([], set())


@pytest.mark.asyncio
async def test_an_escalation_keeps_its_clause_ids_and_drops_its_words(make, tickets, ctx) -> None:
    """The model's reason never enters memory: read back as the customer's
    records, it would ground the model's next claims. Invented ids are dropped."""
    t = tickets["T-001"]
    llm = ScriptedLLM(completion("escalate", {
        "ticket_id": t.ticket_id,
        "reason": "PROSE-SENTINEL: policy is silent (ESC-001.6), see also ZZZ-999.1",
    }))
    await make(llm).run(t)
    [episode] = history(ctx, t.customer_id)
    assert episode.resolution == "Escalated to a human. Clauses cited: ESC-001.6."


@pytest.mark.asyncio
async def test_a_held_refund_is_remembered_as_held(make, tickets, ctx) -> None:
    t = tickets["T-017"]  # $290, owner request: support-manager band
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": t.ticket_id, "amount": 290, "reason": "within the window"}),
        completion("send_reply", {"ticket_id": t.ticket_id, "body": "Your refund has been submitted for approval."}),
    )
    result = await make(llm).run(t)
    assert result.final_state == "PENDING_APPROVAL"
    [episode] = history(ctx, t.customer_id)
    assert episode.resolution == (
        "Waiting on an approver. Refund of $290.00 held for approval. Clauses cited: ESC-001.2."
    )


@pytest.mark.asyncio
async def test_an_issued_refund_is_remembered_as_issued(make, tickets, ctx) -> None:
    t = tickets["T-001"]
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": t.ticket_id, "amount": 29, "reason": "duplicate charge"}),
        completion("send_reply", {"ticket_id": t.ticket_id, "body": "Refunded."}),
    )
    await make(llm).run(t)
    [episode] = history(ctx, t.customer_id)
    assert episode.resolution == (
        "Resolved by the agent; a reply was sent. Refund of $29.00 issued. Clauses cited: ESC-001.1."
    )


@pytest.mark.asyncio
async def test_a_failed_run_is_remembered(make, tickets, ctx) -> None:
    t = tickets["T-001"]
    result = await make(ScriptedLLM(LLMError("provider down"))).run(t)
    assert result.final_state == "FAILED"
    [episode] = history(ctx, t.customer_id)
    assert episode.resolution == "Escalated to a human; the agent could not complete the run."
