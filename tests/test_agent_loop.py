"""The agent loop, driven by a scripted LLM against the real tools.

The LLM is the only fake. Tools run through the real `MCPServer` over the real
fixtures (lexical backend, isolated var dir), so scopes, the audit log and the
tool handlers are exercised as they will be in an eval. The rig (`conftest`)
gives the loop a read-only server and, separately, a write-scoped one that only
the gate dispatches to. Gate behaviour itself is in `test_gate_loop.py`. A scripted LLM is what
makes each CLAUDE.md failure case reachable on demand, deterministically and
for $0; whether the *real* model behaves well is the eval's question, not
these tests'.

The six failure cases CLAUDE.md requires, one test each, are marked
"failure case".
"""

from __future__ import annotations

import asyncio
import json

import pytest

from evals.loader import load_tickets
from evals.run import run_agent
from evals.schema import Outcome, Ticket, TicketLabels
from src.agent.executor import MCPToolExecutor
from src.agent.llm import LLMError, ToolResult, Usage
from src.agent.loop import READ_SCOPES, WRITE_SCOPES
from src.agent.pricing import EmbeddingMeter, Price, SpendCapExceeded, SpendGuard
from src.agent.states import ALLOWED, State
from src.httpjson import RetryPolicy
from src.tools.catalog import CATALOG, Scope
from tests.agent_fakes import FakeClock, Misbehaving, ScriptedLLM, completion, text_only

def reads(server) -> list[dict]:
    """Audit records for the model's read calls. The gate's writes — sent
    replies, filed escalations — share the audit log and are left out."""
    return [r for r in server.audit.records() if not CATALOG[r["tool"]].gated]


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def ticket(tickets) -> Ticket:
    return tickets["T-001"]


# --- the ordinary path -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_covered_reply_resolves_and_is_sent(make, ticket, ctx, server) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("retrieve_policy", {"query": "refund window monthly plan"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "We have looked into it."}),
    )
    result = await make(llm).run(ticket)

    assert result.outcome is Outcome.RESOLVE
    assert result.final_state == "RESOLVE"
    assert [c.tool.value for c in result.tool_calls] == [
        "lookup_customer", "retrieve_policy", "send_reply",
    ]
    assert [r["body"] for r in ctx.store.records("replies")] == ["We have looked into it."]
    audited = server.audit.records()
    assert [r["tool"] for r in audited] == ["lookup_customer", "retrieve_policy", "send_reply"]
    assert {r["run_id"] for r in audited} == {result.run_id}


@pytest.mark.asyncio
async def test_the_model_sees_each_observation(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}),
    )
    await make(llm).run(ticket)
    assert json.loads(llm.last_observation(1))["customer_id"] == ticket.customer_id


@pytest.mark.asyncio
async def test_an_escalation_ends_in_escalate_and_is_filed(make, ticket, ctx) -> None:
    llm = ScriptedLLM(
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "ESC-001.6: policy is silent"}),
    )
    result = await make(llm).run(ticket)
    assert result.outcome is Outcome.ESCALATE
    assert result.final_state == "ESCALATE"
    assert result.cited_policy_ids == ["POL-ESC-001"]
    [filed] = ctx.store.records("escalations")
    assert filed["reason"] == "ESC-001.6: policy is silent"


@pytest.mark.asyncio
async def test_a_small_refund_executes_and_the_run_continues(make, ticket, ctx, trace) -> None:
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": 29, "reason": "duplicate charge"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "Refunded."}),
    )
    result = await make(llm).run(ticket)

    assert result.outcome is Outcome.RESOLVE
    assert json.loads(llm.last_observation(1))["status"] == "recorded"
    assert [r["amount"] for r in ctx.store.records("refunds")] == [29.0]
    gate = [r for r in trace.records(result.run_id) if r["event"] == "gate"]
    assert [(g["tool"], g["verdict"]) for g in gate] == [
        ("issue_refund", "execute"), ("send_reply", "execute"),
    ]


@pytest.mark.asyncio
async def test_cited_clauses_roll_up_to_policy_ids(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("send_reply", {
            "ticket_id": ticket.ticket_id,
            "body": "Per REF-001.3 and CAN-002.2, and again REF-001.1.",
        }),
    )
    result = await make(llm).run(ticket)
    assert result.cited_policy_ids == ["POL-CAN-002", "POL-REF-001"]


# --- what the model is shown -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_model_is_offered_exactly_the_catalog(make, ticket) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    await make(llm).run(ticket)
    assert llm.tools_offered == list(CATALOG)


@pytest.mark.asyncio
async def test_the_labels_never_reach_the_model(make, ticket) -> None:
    """The eval is meaningless if the answer key is in the prompt."""
    leaky = ticket.model_copy(update={
        "labels": TicketLabels(
            outcome=Outcome.PENDING_APPROVAL,
            expected_tools=[],
            relevant_policy_ids=["POL-TRI-001"],
            confidence="medium",
            label_note="ANSWER-KEY-SENTINEL",
        )
    })
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    await make(llm).run(leaky)

    shown = json.dumps([getattr(m, "content", "") or "" for m in llm.seen[0]])
    for secret in ("ANSWER-KEY-SENTINEL", "POL-TRI-001", "pending_approval", "expected_tools"):
        assert secret not in shown
    assert ticket.body in shown
    assert ticket.requester_email in shown


@pytest.mark.asyncio
async def test_a_past_ticket_search_runs_for_the_tickets_customer_only(make, tickets, server) -> None:
    """Omitted or someone else's, the customer_id that runs is the ticket's:
    the guardrail reads search results as this customer's own records."""
    t = tickets["T-020"]  # CUST-1520 has two prior contacts
    llm = ScriptedLLM(
        completion("search_past_tickets", {"query": "slow response outage"}),
        completion("search_past_tickets", {"query": "charge refunded", "customer_id": "CUST-1560"}),
        completion("escalate", {"ticket_id": t.ticket_id, "reason": "r"}),
    )
    result = await make(llm).run(t)

    assert [r["args"]["customer_id"] for r in reads(server)] == [t.customer_id] * 2
    assert [c.args["customer_id"] for c in result.tool_calls[:2]] == [t.customer_id] * 2
    for turn in (1, 2):
        matches = json.loads(llm.last_observation(turn))["matches"]
        assert {m["customer_id"] for m in matches} <= {t.customer_id}


# --- the state machine ---------------------------------------------------------


@pytest.mark.asyncio
async def test_every_transition_is_legal_and_logged_with_the_run_id(make, ticket, trace) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    result = await make(llm).run(ticket)
    events = trace.records(result.run_id)

    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert events[-1]["final_state"] == "RESOLVE"
    hops = [(e["from_state"], e["to_state"]) for e in events if e["event"] == "transition"]
    assert hops[0][0] == "PLAN"
    for (a, b), (c, _) in zip(hops, hops[1:]):
        assert b == c, "transitions must chain"
    for a, b in hops:
        assert State(b) in ALLOWED[State(a)], f"illegal {a} -> {b}"
    assert hops[-1][1] == "RESOLVE"


@pytest.mark.asyncio
async def test_the_trace_records_exactly_what_the_model_saw(make, ticket, trace) -> None:
    """Reconstructing a run means knowing what each observation said, not
    just that it happened — e.g. whether a clause was ever retrieved."""
    llm = ScriptedLLM(
        completion("retrieve_policy", {"query": "annual renewal refund"}),
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}),
    )
    result = await make(llm).run(ticket)
    observations = [e for e in trace.records(result.run_id) if e["event"] == "observation"]

    assert [o["content"] for o in observations] == [
        llm.last_observation(1), llm.last_observation(2), llm.last_observation(3),
    ]
    assert observations[2]["status"] == "duplicate"


@pytest.mark.asyncio
async def test_run_started_records_what_makes_a_run_comparable(make, ticket, trace) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    result = await make(llm).run(ticket)
    started = trace.records(result.run_id)[0]
    for key in ("ticket_id", "model", "prompt_version", "prompt_sha256", "budgets"):
        assert key in started


# --- failure case: tool returns malformed output --------------------------------


@pytest.mark.asyncio
async def test_failure_case_malformed_tool_output(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "account lookup failed"}),
    )
    executor = MCPToolExecutor(Misbehaving(text="<<garbage>>"), timeout=5)
    result = await make(llm, executor=executor).run(ticket)

    assert json.loads(llm.last_observation(1))["error"] == "malformed"
    assert result.final_state == "ESCALATE", "a bad tool narrows options; it does not crash the run"


# --- failure case: tool times out ------------------------------------------------


@pytest.mark.asyncio
async def test_failure_case_tool_timeout(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    executor = MCPToolExecutor(
        Misbehaving(delay=0.3), timeout=0.05, retry=RetryPolicy(1, 0.0, 0.0)
    )
    result = await make(llm, executor=executor).run(ticket)

    assert json.loads(llm.last_observation(1))["error"] == "timeout"
    assert result.final_state == "RESOLVE"


# --- failure case: retrieval returns nothing relevant ---------------------------


@pytest.mark.asyncio
async def test_failure_case_retrieval_returns_nothing(make, ticket) -> None:
    """A real empty result from the real index, not a faked one."""
    llm = ScriptedLLM(
        completion("retrieve_policy", {"query": "zzzqqx nonexistent kumquat protocol"}),
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "no policy covers this"}),
    )
    result = await make(llm).run(ticket)

    seen = json.loads(llm.last_observation(1))
    assert seen["passages"] == []
    assert "do not answer from memory" in seen["note"].lower()
    assert result.final_state == "ESCALATE"


# --- failure case: repeated identical call (loop detection) ----------------------


@pytest.mark.asyncio
async def test_failure_case_repeated_call_is_not_rerun_then_escalates(make, ticket, server) -> None:
    same = {"customer_id": ticket.customer_id}
    llm = ScriptedLLM(
        completion("lookup_customer", same),
        completion("lookup_customer", same),
        completion("lookup_customer", same),
    )
    result = await make(llm, max_repeats=2).run(ticket)

    assert json.loads(llm.last_observation(2))["error"] == "duplicate_call"
    assert len(reads(server)) == 1, "a duplicate is not executed"
    assert result.final_state == "ESCALATE"
    assert "loop" in result.stop_reason


# --- failure case: LLM returns unparseable tool-call JSON ------------------------


@pytest.mark.asyncio
async def test_failure_case_unparseable_arguments_then_recovery(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", '{"customer_id": '),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    result = await make(llm).run(ticket)
    assert json.loads(llm.last_observation(1))["error"] == "invalid_arguments"
    assert result.final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_failure_case_repeated_unparseable_arguments_escalate(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", "not json"),
        completion("lookup_customer", '["a list"]'),
    )
    result = await make(llm, max_invalid_calls=2).run(ticket)
    assert result.final_state == "ESCALATE"
    assert "invalid" in result.stop_reason


@pytest.mark.asyncio
async def test_an_unknown_tool_is_an_invalid_call(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("delete_account", {"customer_id": "x"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    result = await make(llm).run(ticket)
    assert json.loads(llm.last_observation(1))["error"] == "unknown_tool"
    assert result.final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_a_turn_with_no_tool_call_is_an_invalid_call(make, ticket) -> None:
    llm = ScriptedLLM(
        text_only(),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    result = await make(llm).run(ticket)
    assert result.final_state == "RESOLVE"


@pytest.mark.asyncio
async def test_only_the_first_of_several_calls_runs(make, ticket, server) -> None:
    llm = ScriptedLLM(
        completion(
            "lookup_customer", {"customer_id": ticket.customer_id},
            also=[("retrieve_policy", {"query": "refund"})],
        ),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
    )
    await make(llm).run(ticket)

    assert [r["tool"] for r in reads(server)] == ["lookup_customer"]
    results = [m for m in llm.seen[1] if isinstance(m, ToolResult)]
    assert len(results) == 2, "every requested call gets an answer, executed or not"
    assert json.loads(results[1].content)["error"] == "not_executed"


# --- failure case: budget exhausted mid-run ---------------------------------------


@pytest.mark.asyncio
async def test_failure_case_step_budget_exhausted(make, ticket, server) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("retrieve_policy", {"query": "refund"}),
        completion("retrieve_policy", {"query": "refund window"}),
    )
    result = await make(llm, max_steps=2).run(ticket)

    assert result.final_state == "ESCALATE"
    assert "step budget" in result.stop_reason
    assert len(reads(server)) == 2


@pytest.mark.asyncio
async def test_token_budget_exhausted(make, ticket) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}, tokens=(100, 20)),
        completion("retrieve_policy", {"query": "refund"}, tokens=(100, 20)),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}, tokens=(100, 20)),
    )
    result = await make(llm, max_tokens=250).run(ticket)
    assert result.final_state == "ESCALATE"
    assert "token budget" in result.stop_reason


@pytest.mark.asyncio
async def test_wall_clock_budget_exhausted(make, ticket) -> None:
    clock = FakeClock()
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("retrieve_policy", {"query": "refund"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "ok"}),
        clock=clock,
        advance=60.0,
    )
    result = await make(llm, clock=clock, max_seconds=100).run(ticket)
    assert result.final_state == "ESCALATE"
    assert "wall-clock" in result.stop_reason


@pytest.mark.asyncio
async def test_a_hung_llm_call_is_cut_off_by_the_wall_clock(make, ticket) -> None:
    async def hang():
        await asyncio.sleep(5)

    result = await make(ScriptedLLM(hang), max_seconds=0.2).run(ticket)
    assert result.final_state == "ESCALATE"
    assert "wall-clock" in result.stop_reason


# --- the provider failing ----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unavailable_llm_ends_in_failed_reported_as_escalate(make, ticket, trace) -> None:
    """FAILED is distinct in the trace, so it can be counted; in the result
    it is an escalation, because a ticket the agent could not work still has
    to reach a human."""
    llm = ScriptedLLM(LLMError("chat request rejected, HTTP 401: bad key"))
    result = await make(llm).run(ticket)

    assert result.final_state == "FAILED"
    assert result.outcome is Outcome.ESCALATE
    assert "401" in result.stop_reason
    assert result.tool_calls == []


# --- spend --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_spend_cap_aborts_the_run_loudly(make, ticket) -> None:
    """Not an escalation: an eval that runs out of money must stop, not report
    partial numbers as if they were complete."""
    guard = SpendGuard(cap_usd=0.00001, price=Price(input_per_mtok=0.15, output_per_mtok=0.60))
    llm = ScriptedLLM(completion("lookup_customer", {"customer_id": ticket.customer_id}))
    with pytest.raises(SpendCapExceeded):
        await make(llm, spend=guard).run(ticket)


def test_pricing_is_per_million_tokens() -> None:
    price = Price(input_per_mtok=0.15, output_per_mtok=0.60)
    assert price.cost(Usage(1_000_000, 1_000_000)) == pytest.approx(0.75)


PRICE = Price(input_per_mtok=0.15, output_per_mtok=0.60)
#: One scripted turn: 100 input and 20 output tokens.
TURN_USD = (100 * 0.15 + 20 * 0.60) / 1_000_000


@pytest.mark.asyncio
async def test_the_result_carries_cost_and_wall_clock(make, ticket) -> None:
    clock = FakeClock()
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}),
        clock=clock, advance=1.5,
    )
    result = await make(llm, clock=clock, spend=SpendGuard(cap_usd=1.0, price=PRICE)).run(ticket)
    assert result.cost_usd == pytest.approx(2 * TURN_USD)
    assert result.elapsed_s == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_without_a_price_the_cost_is_unknown_not_zero(make, ticket) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    result = await make(llm).run(ticket)
    assert result.cost_usd is None
    assert result.elapsed_s is not None


class FakeEmbedder:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.tokens_used = 0


@pytest.mark.asyncio
async def test_embedding_spend_during_a_run_is_charged_to_it(make, ticket, trace) -> None:
    """Tokens the provider counted while this ticket ran are this ticket's;
    tokens spent building an index beforehand are no ticket's."""
    embedder = FakeEmbedder()
    embedder.tokens_used = 5_000  # the index build, before any ticket
    meter = EmbeddingMeter([embedder])

    async def embed_then_escalate():
        embedder.tokens_used += 1_000_000  # $0.02 at list price
        return completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"})

    guard = SpendGuard(cap_usd=1.0, price=PRICE)
    result = await make(ScriptedLLM(embed_then_escalate), spend=guard, embedding_meter=meter).run(ticket)

    assert result.cost_usd == pytest.approx(TURN_USD + 0.02)
    assert guard.spent_usd == pytest.approx(TURN_USD + 0.02)
    [finished] = [e for e in trace.records(result.run_id) if e["event"] == "run_finished"]
    assert finished["cost_usd"] == pytest.approx(TURN_USD)
    assert finished["embedding_cost_usd"] == pytest.approx(0.02)


def test_an_embedding_model_without_a_price_is_refused() -> None:
    with pytest.raises(ValueError, match="text-embedding-unknown"):
        EmbeddingMeter([FakeEmbedder("text-embedding-unknown")])


def test_lexical_tools_have_no_embedders(ctx) -> None:
    assert ctx.embedders() == []


# --- wiring ---------------------------------------------------------------------------


def test_the_loop_is_granted_read_scopes_only() -> None:
    """The model's calls go to a read-only server; writes go to a separate
    server that only the gate dispatches to."""
    assert READ_SCOPES == {Scope.CUSTOMER_READ, Scope.TICKETS_READ, Scope.POLICY_READ}
    assert WRITE_SCOPES == {Scope.REFUND_WRITE, Scope.REPLY_WRITE, Scope.ESCALATION_WRITE}


@pytest.mark.asyncio
async def test_the_loop_runs_under_the_eval_harness(make, ticket) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    results = await run_agent(make(llm), [ticket])
    assert results[ticket.ticket_id].outcome is Outcome.ESCALATE
