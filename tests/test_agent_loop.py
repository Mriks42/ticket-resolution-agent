"""The agent loop, driven by a scripted LLM against the real tools.

The LLM is the only fake. Tools run through the real `MCPServer` over the real
fixtures (lexical backend, isolated var dir), so scopes, the audit log and the
tool handlers are exercised as they will be in an eval. A scripted LLM is what
makes each CLAUDE.md failure case reachable on demand, deterministically and
for $0; whether the *real* model behaves well is the eval's question, not
these tests'.

The six failure cases CLAUDE.md requires, one test each, are marked
"failure case".
"""

from __future__ import annotations

import asyncio
import json
import time
from itertools import count

import pytest

from evals.loader import load_tickets
from evals.run import run_agent
from evals.schema import Outcome, Ticket, TicketLabels
from src.agent.config import AgentConfig
from src.agent.executor import MCPToolExecutor
from src.agent.llm import Completion, LLMError, ToolCallRequest, ToolResult, Usage
from src.agent.loop import READ_SCOPES, AgentLoop
from src.agent.pricing import Price, SpendCapExceeded, SpendGuard
from src.agent.states import ALLOWED, State
from src.agent.trace import TraceLog
from src.httpjson import RetryPolicy
from src.mcp.server import MCPServer
from src.tools.catalog import CATALOG, Scope
from src.tools.context import ToolContext
from src.tools.retrieval import RetrievalConfig

_ids = count(1)


def completion(name, args, *, tokens=(100, 20), also=()) -> Completion:
    """One model turn calling `name`. `args` may be a dict or a raw string."""

    def request(n, a):
        raw = a if isinstance(a, str) else json.dumps(a)
        return ToolCallRequest(call_id=f"call_{next(_ids)}", name=n, arguments=raw)

    calls = (request(name, args), *(request(n, a) for n, a in also))
    return Completion(
        text=None,
        tool_calls=calls,
        usage=Usage(*tokens),
        served_model="scripted-2026-10-14",
        finish_reason="tool_calls",
    )


def text_only(text="Let me think about this.") -> Completion:
    return Completion(text=text, tool_calls=(), usage=Usage(100, 20), served_model="scripted")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ScriptedLLM:
    """Plays back completions in order and records what it was shown."""

    model = "scripted"

    def __init__(self, *steps, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        self.steps = list(steps)
        self.seen: list[list] = []
        self.tools_offered: list[str] = []
        self.clock = clock
        self.advance = advance

    async def complete(self, messages, tools):
        self.seen.append(list(messages))
        self.tools_offered = [t.name for t in tools]
        if self.clock is not None:
            self.clock.now += self.advance
        if not self.steps:
            raise AssertionError("scripted LLM ran out of steps")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return await step()
        return step

    def last_observation(self, turn: int) -> str:
        """The tool result the model saw at the start of `turn` (0-based)."""
        results = [m for m in self.seen[turn] if isinstance(m, ToolResult)]
        return results[-1].content


class Misbehaving:
    def __init__(self, *, text: str = "{}", delay: float = 0.0) -> None:
        self.text, self.delay = text, delay

    def handle(self, request: dict) -> dict:
        if self.delay:
            time.sleep(self.delay)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"content": [{"type": "text", "text": self.text}], "isError": False},
        }


@pytest.fixture(scope="module")
def tickets() -> dict[str, Ticket]:
    return {t.ticket_id: t for t in load_tickets()}


@pytest.fixture
def ticket(tickets) -> Ticket:
    return tickets["T-001"]


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext.build(var_dir=tmp_path / "var", retrieval=RetrievalConfig())


@pytest.fixture
def server(ctx, tmp_path) -> MCPServer:
    return MCPServer(granted_scopes=READ_SCOPES, context=ctx, var_dir=tmp_path / "var")


@pytest.fixture
def trace(tmp_path) -> TraceLog:
    return TraceLog(tmp_path / "var" / "trace" / "trace.jsonl")


@pytest.fixture
def make(server, trace):
    def build(llm, *, executor=None, clock=time.monotonic, spend=None, **config) -> AgentLoop:
        return AgentLoop(
            llm=llm,
            executor=executor or MCPToolExecutor(server, timeout=5),
            trace=trace,
            config=AgentConfig(**config),
            clock=clock,
            spend=spend,
        )

    return build


# --- the ordinary path -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reply_resolves_and_is_held_not_sent(make, ticket, ctx, server) -> None:
    llm = ScriptedLLM(
        completion("lookup_customer", {"customer_id": ticket.customer_id}),
        completion("retrieve_policy", {"query": "refund window monthly plan"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "Refunded under REF-001.1."}),
    )
    result = await make(llm).run(ticket)

    assert result.outcome is Outcome.RESOLVE
    assert result.final_state == "RESOLVE"
    assert [c.tool.value for c in result.tool_calls] == [
        "lookup_customer", "retrieve_policy", "send_reply",
    ]
    assert ctx.store.records("replies") == [], "gated calls are proposals, never executed"
    audited = server.audit.records()
    assert [r["tool"] for r in audited] == ["lookup_customer", "retrieve_policy"]
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
async def test_an_escalation_ends_in_escalate(make, ticket, ctx) -> None:
    llm = ScriptedLLM(
        completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "ESC-001.6: policy is silent"}),
    )
    result = await make(llm).run(ticket)
    assert result.outcome is Outcome.ESCALATE
    assert result.final_state == "ESCALATE"
    assert result.cited_policy_ids == ["POL-ESC-001"]
    assert ctx.store.records("escalations") == []


@pytest.mark.asyncio
async def test_a_refund_is_held_and_the_run_continues(make, ticket, ctx, trace) -> None:
    llm = ScriptedLLM(
        completion("issue_refund", {"ticket_id": ticket.ticket_id, "amount": 29, "reason": "REF-001.1"}),
        completion("send_reply", {"ticket_id": ticket.ticket_id, "body": "Submitted."}),
    )
    result = await make(llm).run(ticket)

    assert result.outcome is Outcome.RESOLVE
    held = json.loads(llm.last_observation(1))
    assert held["status"] == "held"
    assert "not" in held["message"] and "executed" in held["message"]
    assert ctx.store.records("refunds") == []
    proposals = [r for r in trace.records(result.run_id) if r["event"] == "proposal"]
    assert [p["tool"] for p in proposals] == ["issue_refund", "send_reply"]
    assert all(p["held"] for p in proposals)


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
    assert len(server.audit.records()) == 1, "a duplicate is not executed"
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

    assert [r["tool"] for r in server.audit.records()] == ["lookup_customer"]
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
    assert len(server.audit.records()) == 2


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


# --- wiring ---------------------------------------------------------------------------


def test_the_loop_is_granted_read_scopes_only() -> None:
    assert READ_SCOPES == {Scope.CUSTOMER_READ, Scope.TICKETS_READ, Scope.POLICY_READ}


@pytest.mark.asyncio
async def test_the_loop_runs_under_the_eval_harness(make, ticket) -> None:
    llm = ScriptedLLM(completion("escalate", {"ticket_id": ticket.ticket_id, "reason": "r"}))
    results = await run_agent(make(llm), [ticket])
    assert results[ticket.ticket_id].outcome is Outcome.ESCALATE
