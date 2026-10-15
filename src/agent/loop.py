"""The agent loop: an explicit state machine, one tool call per step.

    PLAN     ask the model for exactly one tool call, given everything observed
    ACT      validate it, then run it (a read) or hold it (a gated action)
    OBSERVE  hand the result back to the model, and plan again

The transitions are the table in `src/agent/states.py`; every hop is checked
against it and written to the trace with the run_id.

**Gated calls go through the HITL gate** (`src/agent/gate.py`, ADR 0006).
The model's own calls reach a server granted read scopes only. A gated call is
decided by the gate's deterministic rules and then carried out by the loop:

- EXECUTE   dispatched to a separate write-scoped server that only the gate
            reaches. An executed reply ends the run in RESOLVE.
- HOLD      a refund waits for its approver and the run continues, so the
            model can write the reply; that reply is held with it, and the
            ticket ends in PENDING_APPROVAL.
- ESCALATE  the action does not run; the ticket ends in ESCALATE.

`escalate` itself always goes through, and wins over anything held: the gate
never overrides a hand-off. Every run that ends in ESCALATE or FAILED files
exactly one escalation, by the model or by the system, so no stopped ticket is
lost. Anything held is attached to it.

**What the loop stops, and how.** Each of these ends the run in ESCALATE with
a `stop_reason`, never in a silent failure:

- the step, token or wall-clock budget runs out (`AgentConfig`)
- the model repeats an identical call `max_repeats` times. The first repeat is
  not re-run; the model is told it already has the result.
- the model makes `max_invalid_calls` invalid calls: unparseable arguments,
  an unknown tool, or a turn with no tool call. The first one is reported back
  so the model can correct itself.

FAILED is reserved for the one thing the loop cannot work around: no usable
completion from the model at all. It is reported as an escalation in the
result, because the ticket still has to reach a human, and kept distinct in
`final_state` and the trace so it can be counted.

**Exceeding the spend cap is not handled here.** `SpendCapExceeded` propagates
out of `run()` on purpose, so an eval that runs out of money stops instead of
reporting partial numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from evals.schema import AgentResult, Outcome, Ticket, Tool, ToolCall
from src.agent.config import AgentConfig
from src.agent.executor import MCPToolExecutor, Observation, ToolExecutor
from src.agent.gate import Decision, Verdict, cited_clause_ids, decide_refund, decide_reply
from src.agent.llm import (
    Assistant,
    Completion,
    FunctionSchema,
    LLMClient,
    LLMError,
    Message,
    System,
    ToolCallRequest,
    ToolResult,
    User,
)
from src.agent.openai_chat import OpenAIChatClient
from src.agent.pricing import SpendGuard
from src.agent.prompt import Prompt, load_prompt
from src.agent.states import TERMINAL, State, check
from src.agent.trace import TraceLog
from src.mcp.server import MCPServer
from src.tools.catalog import CATALOG
from src.tools.context import ToolContext
from src.tools.policy import Clause, load_policy_index
from src.tools.retrieval import RetrievalConfig

READ_SCOPES = frozenset(spec.scope for spec in CATALOG.values() if not spec.gated)
#: Granted to the server only the gate dispatches to. The model never reaches it.
WRITE_SCOPES = frozenset(spec.scope for spec in CATALOG.values() if spec.gated)
GATED = frozenset(name for name, spec in CATALOG.items() if spec.gated)

OUTCOME = {
    State.RESOLVE: Outcome.RESOLVE,
    State.ESCALATE: Outcome.ESCALATE,
    State.PENDING_APPROVAL: Outcome.PENDING_APPROVAL,
    State.FAILED: Outcome.ESCALATE,
}

#: Per eval run, in USD. Override with EVAL_SPEND_CAP_USD.
DEFAULT_SPEND_CAP_USD = 1.00

PENDING_MESSAGE = (
    "Submitted for {approver} approval under {clause}; not issued yet. Tell the customer "
    "it has been submitted for approval, not that it is done."
)
EMPTY_RETRIEVAL_NOTE = (
    "No policy passage matched this query closely enough to be returned. Do not answer "
    "from memory: search with different words, or escalate if no policy covers the situation."
)

_CLAUSE_ID = re.compile(r"\b([A-Z]{3}-\d{3})\.\d+\b")


def tool_schemas() -> list[FunctionSchema]:
    """Every catalog tool, as the model sees it. The catalog is the one source."""
    return [FunctionSchema(s.name, s.description, s.input_schema) for s in CATALOG.values()]


def render_ticket(ticket: Ticket) -> str:
    """The ticket as the model sees it — these four fields and nothing else.
    `ticket.labels` is the answer key and must never reach the prompt."""
    return (
        f"Ticket {ticket.ticket_id}\n"
        f"Customer ID: {ticket.customer_id}\n"
        f"From: {ticket.requester_email}\n\n"
        f"{ticket.body}"
    )


def cited_policy_ids(texts: Iterable[str]) -> list[str]:
    """Clause ids found in proposal text, rolled up to their documents."""
    return sorted({f"POL-{m}" for text in texts for m in _CLAUSE_ID.findall(text)})


def _error(kind: str, message: str) -> str:
    return json.dumps({"error": kind, "message": message})


def _positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


@dataclass
class _Run:
    run_id: str
    ticket: Ticket
    started: float
    state: State = State.PLAN
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    invalid: int = 0
    repeats: int = 0
    stop_reason: str | None = None
    seen: set[str] = field(default_factory=set)
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    proposal_texts: list[str] = field(default_factory=list)
    #: The call chosen at PLAN, and any extra calls the model sent with it.
    current: ToolCallRequest | None = None
    extras: tuple[ToolCallRequest, ...] = ()
    #: Messages to hand the model at OBSERVE.
    pending: list[Message] = field(default_factory=list)
    #: Clause ids the model has been shown in this run, by retrieval or by the
    #: gate. A citation outside this set is a hallucinated-policy event.
    retrieved: set[str] = field(default_factory=set)
    #: Gated actions waiting for an approver.
    held: list[dict] = field(default_factory=list)
    #: Refunds on this ticket so far, executed or held. The gate bands the total.
    refunded: float = 0.0
    escalation_filed: bool = False
    owner_email: str | None = None
    owner_checked: bool = False


class AgentLoop:
    name = "agent-loop"

    def __init__(
        self,
        *,
        llm: LLMClient,
        executor: ToolExecutor,
        writer: ToolExecutor,
        trace: TraceLog,
        config: AgentConfig = AgentConfig(),
        prompt: Prompt | None = None,
        clock: Callable[[], float] = time.monotonic,
        spend: SpendGuard | None = None,
        context: dict | None = None,
        clause_lookup: Callable[[str], Clause | None] | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._writer = writer
        self._clause_lookup = clause_lookup or load_policy_index().get
        self._trace = trace
        self._config = config
        self._prompt = prompt or load_prompt()
        self._clock = clock
        self._spend = spend
        self._context = context or {}
        self._schemas = tool_schemas()

    async def run(self, ticket: Ticket) -> AgentResult:
        run = _Run(run_id=f"run-{uuid.uuid4().hex[:12]}", ticket=ticket, started=self._clock())
        run.messages = [System(self._prompt.text), User(render_ticket(ticket))]
        self._trace.emit(
            run.run_id,
            "run_started",
            ticket_id=ticket.ticket_id,
            customer_id=ticket.customer_id,
            model=self._llm.model,
            temperature=self._config.temperature,
            seed=self._config.seed,
            prompt_version=self._prompt.version,
            prompt_sha256=self._prompt.sha256,
            budgets=self._config.budgets(),
            context=self._context,
        )

        while run.state not in TERMINAL:
            if run.state is State.PLAN:
                await self._plan(run)
            elif run.state is State.ACT:
                await self._act(run)
            else:
                self._observe(run)

        if run.state in (State.ESCALATE, State.FAILED) and not run.escalation_filed:
            await self._file_escalation(run, run.stop_reason or "stopped", filed_by="system")

        self._trace.emit(
            run.run_id,
            "run_finished",
            final_state=run.state.value,
            outcome=OUTCOME[run.state].value,
            stop_reason=run.stop_reason,
            steps=run.steps,
            tokens=run.tokens,
            cost_usd=round(run.cost_usd, 6),
            elapsed_s=round(self._clock() - run.started, 3),
        )
        return AgentResult(
            ticket_id=ticket.ticket_id,
            run_id=run.run_id,
            outcome=OUTCOME[run.state],
            tool_calls=run.tool_calls,
            cited_policy_ids=cited_policy_ids(run.proposal_texts),
            final_state=run.state.value,
            stop_reason=run.stop_reason,
        )

    # --- states ------------------------------------------------------------

    async def _plan(self, run: _Run) -> None:
        cfg = self._config
        if self._remaining(run) <= 0:
            return self._go(run, State.ESCALATE, f"wall-clock budget exhausted ({cfg.max_seconds:g}s)")

        started = time.perf_counter()
        try:
            completion = await asyncio.wait_for(
                self._llm.complete(run.messages, self._schemas), timeout=self._remaining(run)
            )
        except TimeoutError:
            return self._go(
                run, State.ESCALATE,
                f"wall-clock budget exhausted ({cfg.max_seconds:g}s) waiting for the model",
            )
        except LLMError as exc:
            return self._go(run, State.FAILED, f"no usable completion from the model: {exc}")

        self._account(run, completion, started)
        if run.tokens > cfg.max_tokens:
            return self._go(
                run, State.ESCALATE, f"token budget exhausted ({run.tokens} of {cfg.max_tokens})"
            )

        run.messages.append(Assistant(completion.text, completion.tool_calls))
        run.current = completion.tool_calls[0] if completion.tool_calls else None
        run.extras = completion.tool_calls[1:]
        chose = run.current.name if run.current else "no tool"
        self._go(run, State.ACT, f"model chose {chose}")

    async def _act(self, run: _Run) -> None:
        cfg, call = self._config, run.current
        run.pending = [
            ToolResult(
                extra.call_id,
                _error(
                    "not_executed",
                    "Only one tool call runs per turn; this one was ignored. "
                    "Call it again if you still need it.",
                ),
            )
            for extra in run.extras
        ]

        if call is None:
            return self._invalid(run, None, "no_tool_call", "Every turn must call exactly one tool.")
        if call.name not in CATALOG:
            return self._invalid(
                run, call, "unknown_tool",
                f"There is no tool named {call.name!r}. Use one of: {', '.join(CATALOG)}.",
            )
        try:
            args = json.loads(call.arguments)
        except ValueError:
            args = None
        if not isinstance(args, dict):
            return self._invalid(
                run, call, "invalid_arguments",
                "Arguments must be a JSON object matching the tool's schema.",
            )

        if call.name == "issue_refund" and not _positive_number(args.get("amount")):
            return self._invalid(
                run, call, "invalid_arguments", "amount must be a positive number of US dollars."
            )

        key = f"{call.name}:{json.dumps(args, sort_keys=True)}"
        if key in run.seen:
            run.repeats += 1
            if run.repeats >= cfg.max_repeats:
                return self._go(
                    run, State.ESCALATE,
                    f"loop detected: {call.name} repeated with identical arguments",
                )
            return self._respond(
                run, call, args, "duplicate",
                _error(
                    "duplicate_call",
                    "Identical to an earlier call, so it was not run again. Its result "
                    "is already in this conversation: use it, or do something different.",
                ),
                f"duplicate {call.name} not executed",
            )

        if run.steps >= cfg.max_steps:
            return self._go(
                run, State.ESCALATE, f"step budget exhausted ({run.steps} of {cfg.max_steps})"
            )
        run.steps += 1
        run.seen.add(key)
        run.tool_calls.append(ToolCall(tool=Tool(call.name), args=args))

        if call.name in GATED:
            run.proposal_texts.extend(str(v) for v in args.values())
            return await self._gated(run, call, args)

        try:
            obs = await asyncio.wait_for(
                self._executor.call(call.name, args, run_id=run.run_id),
                timeout=max(self._remaining(run), 0.001),
            )
        except TimeoutError:
            return self._go(
                run, State.ESCALATE,
                f"wall-clock budget exhausted ({cfg.max_seconds:g}s) during {call.name}",
            )
        content = obs.for_model()
        if call.name == "retrieve_policy" and obs.status == "ok":
            passages = obs.payload.get("passages") or []
            run.retrieved |= {p["clause_id"] for p in passages if isinstance(p, dict) and "clause_id" in p}
            if not passages:
                content = json.dumps({**obs.payload, "note": EMPTY_RETRIEVAL_NOTE})
        self._respond(
            run, call, args, obs.status, content, f"{call.name} {obs.status}",
            latency_ms=obs.latency_ms, attempts=obs.attempts,
        )

    async def _gated(self, run: _Run, call: ToolCallRequest, args: dict) -> None:
        if call.name == "escalate":
            self._gate_event(run, call, args, Decision(
                Verdict.EXECUTE, "handoff", "escalations always go through and win over anything held",
            ))
            await self._file_escalation(
                run, str(args.get("reason") or ""), filed_by="model", model_trace=args.get("trace"),
            )
            return self._go(run, State.ESCALATE, "escalate proposed")

        facts: dict = {}
        if call.name == "issue_refund":
            owner = await self._owner_email(run)
            facts = {
                "requester_email": run.ticket.requester_email,
                "owner_verified": owner is not None,
                "refunded_before": run.refunded,
            }
            decision = decide_refund(
                amount=args["amount"],
                already_refunded=run.refunded,
                requester_email=run.ticket.requester_email,
                owner_email=owner,
                cited=cited_clause_ids(str(args.get("reason", ""))),
                retrieved=run.retrieved,
            )
        else:
            decision = decide_reply(
                cited=cited_clause_ids(str(args.get("body", ""))),
                retrieved=run.retrieved,
                pending=bool(run.held),
            )
        self._gate_event(run, call, args, decision, facts)

        if decision.verdict is Verdict.ESCALATE:
            prefix = f"{decision.clause}: " if decision.clause else ""
            return self._go(
                run, State.ESCALATE, f"{prefix}{call.name} blocked by the gate, {decision.detail}"
            )

        if decision.verdict is Verdict.HOLD:
            if call.name == "send_reply":
                run.held.append({"tool": call.name, "args": args, "rule": decision.rule})
                return self._go(run, State.PENDING_APPROVAL, "reply held with the pending approval")
            run.held.append({
                "tool": call.name, "args": args,
                "approver": decision.approver, "clause": decision.clause,
            })
            run.refunded += args["amount"]
            content = json.dumps({
                "status": "pending_approval",
                "approver": decision.approver,
                "message": PENDING_MESSAGE.format(approver=decision.approver, clause=decision.clause),
                "policy": self._show_clause(run, decision.clause),
            })
            return self._respond(
                run, call, args, "pending_approval", content,
                f"{call.name} held for {decision.approver} approval",
            )

        obs = await self._writer.call(call.name, args, run_id=run.run_id)
        if call.name == "issue_refund" and obs.status == "timeout":
            return self._go(
                run, State.ESCALATE,
                "refund outcome unknown: issue_refund timed out and may or may not have executed",
            )
        if call.name == "send_reply":
            if obs.status != "ok":
                return self._go(
                    run, State.ESCALATE, f"the reply could not be sent ({obs.status}: {obs.message})"
                )
            return self._go(run, State.RESOLVE, "reply sent")
        if call.name == "issue_refund" and obs.status == "ok":
            run.refunded += args["amount"]
        content = obs.for_model()
        if obs.status == "ok" and decision.clause:
            content = json.dumps({**obs.payload, "policy": self._show_clause(run, decision.clause)})
        self._respond(
            run, call, args, obs.status, content, f"{call.name} executed ({obs.status})",
            latency_ms=obs.latency_ms, attempts=obs.attempts,
        )

    async def _owner_email(self, run: _Run) -> str | None:
        """The verified owner, from the gate's own audited lookup — never from
        anything the model said. None if it cannot be established."""
        if not run.owner_checked:
            obs = await self._executor.call(
                "lookup_customer", {"customer_id": run.ticket.customer_id}, run_id=run.run_id
            )
            run.owner_checked = True
            owner = obs.payload.get("owner_email") if obs.status == "ok" else None
            run.owner_email = owner if isinstance(owner, str) and owner.strip() else None
        return run.owner_email

    def _show_clause(self, run: _Run, clause_id: str | None) -> dict | None:
        """A clause the gate decided on, with its text. Shown to the model, so
        it counts as retrieved in this run and can be cited."""
        clause = self._clause_lookup(clause_id) if clause_id else None
        if clause is None:
            return None
        run.retrieved.add(clause.clause_id)
        return {"clause_id": clause.clause_id, "text": clause.text}

    async def _file_escalation(
        self, run: _Run, reason: str, *, filed_by: str, model_trace=None
    ) -> Observation:
        trace = [item for item in model_trace if isinstance(item, dict)] if isinstance(model_trace, list) else []
        trace.append({"run_id": run.run_id, "filed_by": filed_by})
        if run.held:
            trace.append({"held_for_approval": run.held})
        obs = await self._writer.call(
            "escalate",
            {"ticket_id": run.ticket.ticket_id, "reason": reason or "no reason given", "trace": trace},
            run_id=run.run_id,
        )
        run.escalation_filed = True
        self._trace.emit(
            run.run_id, "escalation_filed",
            filed_by=filed_by, reason=reason, status=obs.status, message=obs.message,
        )
        return obs

    def _gate_event(
        self, run: _Run, call: ToolCallRequest, args: dict, decision: Decision, facts: dict | None = None
    ) -> None:
        self._trace.emit(
            run.run_id, "gate",
            tool=call.name, args=args, verdict=decision.verdict.value, rule=decision.rule,
            clause=decision.clause, approver=decision.approver, detail=decision.detail,
            facts=facts or {},
        )

    def _observe(self, run: _Run) -> None:
        run.messages.extend(run.pending)
        run.pending, run.current, run.extras = [], None, ()
        self._go(run, State.PLAN, "observation recorded")

    # --- helpers -------------------------------------------------------------

    def _invalid(self, run: _Run, call: ToolCallRequest | None, kind: str, message: str) -> None:
        run.invalid += 1
        if run.invalid >= self._config.max_invalid_calls:
            return self._go(
                run, State.ESCALATE,
                f"model made {run.invalid} invalid tool calls (last: {kind})",
            )
        if call is None:
            # No call means no call id to answer, so the correction goes in as
            # a user turn instead of a tool result.
            return self._respond(run, None, None, kind, f"[{kind}] {message}", f"invalid call: {kind}")
        self._respond(run, call, None, kind, _error(kind, message), f"invalid call: {kind}")

    def _respond(
        self,
        run: _Run,
        call: ToolCallRequest | None,
        args: dict | None,
        status: str,
        content: str,
        reason: str,
        *,
        latency_ms: float = 0.0,
        attempts: int = 0,
    ) -> None:
        """Queue what the model will see, record it verbatim in the trace, and
        move to OBSERVE. The one place an observation is made, so the trace
        cannot drift from what the model was shown."""
        self._trace.emit(
            run.run_id, "observation",
            tool=call.name if call else None, args=args, status=status,
            content=content, latency_ms=latency_ms, attempts=attempts,
        )
        run.pending.insert(0, ToolResult(call.call_id, content) if call else User(content))
        self._go(run, State.OBSERVE, reason)

    def _account(self, run: _Run, completion: Completion, started: float) -> None:
        run.turns += 1
        run.tokens += completion.usage.total
        cost = None
        if self._spend is not None:
            cost = self._spend.charge(completion.usage)  # may raise SpendCapExceeded
            run.cost_usd += cost
        self._trace.emit(
            run.run_id, "llm_call",
            turn=run.turns,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            served_model=completion.served_model,
            finish_reason=completion.finish_reason,
            tool_calls=[c.name for c in completion.tool_calls],
            cost_usd=None if cost is None else round(cost, 8),
        )

    def _go(self, run: _Run, target: State, reason: str) -> None:
        check(run.state, target)
        self._trace.emit(
            run.run_id, "transition",
            from_state=run.state.value, to_state=target.value, reason=reason, step=run.steps,
        )
        run.state = target
        if target in TERMINAL:
            run.stop_reason = reason

    def _remaining(self, run: _Run) -> float:
        return self._config.max_seconds - (self._clock() - run.started)


class LoopAgent(AgentLoop):
    """`AgentLoop` wired from the environment, for the eval harness:

        python -m evals.run --agent src.agent.loop:LoopAgent

    The entry point loads `.env`; this reads `os.environ` only.
    """

    def __init__(self) -> None:
        config = AgentConfig.from_env()
        retrieval = RetrievalConfig.from_env()
        context = ToolContext.build(retrieval=retrieval)
        reader = MCPServer(granted_scopes=READ_SCOPES, context=context, var_dir=context.var_dir)
        writer = MCPServer(granted_scopes=WRITE_SCOPES, context=context, var_dir=context.var_dir)
        cap = float(os.environ.get("EVAL_SPEND_CAP_USD") or DEFAULT_SPEND_CAP_USD)
        super().__init__(
            llm=OpenAIChatClient(model=config.model, temperature=config.temperature, seed=config.seed),
            executor=MCPToolExecutor(reader, timeout=config.tool_timeout),
            writer=MCPToolExecutor.for_writes(writer, timeout=config.tool_timeout),
            clause_lookup=context.fixtures.policy.get,
            trace=TraceLog.in_var_dir(context.var_dir),
            config=config,
            spend=SpendGuard.for_model(config.model, cap),
            context={"retrieval": retrieval.describe(), "spend_cap_usd": cap},
        )
