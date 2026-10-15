"""The agent loop: an explicit state machine, one tool call per step.

    PLAN     ask the model for exactly one tool call, given everything observed
    ACT      validate it, then run it (a read) or hold it (a gated action)
    OBSERVE  hand the result back to the model, and plan again

The transitions are the table in `src/agent/states.py`; every hop is checked
against it and written to the trace with the run_id.

**Gated calls are held, never executed.** `issue_refund`, `send_reply` and
`escalate` are recorded as proposals. A held refund lets the run continue; a
held reply ends it in RESOLVE and a held escalation in ESCALATE. The HITL gate,
a later session, is what turns proposals into executions or PENDING_APPROVAL;
until it lands, PENDING_APPROVAL is unreachable. The MCP server the loop talks
to is granted read scopes only, so a gated call that reached it by mistake
would be refused and audited.

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
from src.agent.executor import MCPToolExecutor, ToolExecutor
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
from src.tools.retrieval import RetrievalConfig

READ_SCOPES = frozenset(spec.scope for spec in CATALOG.values() if not spec.gated)
GATED = frozenset(name for name, spec in CATALOG.items() if spec.gated)

#: Gated calls that end the run when proposed, and the state each ends it in.
FINISHING = {"send_reply": State.RESOLVE, "escalate": State.ESCALATE}

OUTCOME = {
    State.RESOLVE: Outcome.RESOLVE,
    State.ESCALATE: Outcome.ESCALATE,
    State.PENDING_APPROVAL: Outcome.PENDING_APPROVAL,
    State.FAILED: Outcome.ESCALATE,
}

#: Per eval run, in USD. Override with EVAL_SPEND_CAP_USD.
DEFAULT_SPEND_CAP_USD = 1.00

HELD_MESSAGE = (
    "Recorded for approval; not executed. Do not tell the customer it has been completed."
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


class AgentLoop:
    name = "agent-loop"

    def __init__(
        self,
        *,
        llm: LLMClient,
        executor: ToolExecutor,
        trace: TraceLog,
        config: AgentConfig = AgentConfig(),
        prompt: Prompt | None = None,
        clock: Callable[[], float] = time.monotonic,
        spend: SpendGuard | None = None,
        context: dict | None = None,
    ) -> None:
        self._llm = llm
        self._executor = executor
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
            self._trace.emit(run.run_id, "proposal", tool=call.name, args=args, held=True)
            if call.name in FINISHING:
                return self._go(run, FINISHING[call.name], f"{call.name} proposed, held for approval")
            return self._respond(
                run, call, args, "held",
                json.dumps({"status": "held", "message": HELD_MESSAGE}),
                f"{call.name} held for approval",
            )

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
        if call.name == "retrieve_policy" and obs.status == "ok" and not obs.payload.get("passages"):
            content = json.dumps({**obs.payload, "note": EMPTY_RETRIEVAL_NOTE})
        self._respond(
            run, call, args, obs.status, content, f"{call.name} {obs.status}",
            latency_ms=obs.latency_ms, attempts=obs.attempts,
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
        server = MCPServer(granted_scopes=READ_SCOPES, context=context, var_dir=context.var_dir)
        cap = float(os.environ.get("EVAL_SPEND_CAP_USD") or DEFAULT_SPEND_CAP_USD)
        super().__init__(
            llm=OpenAIChatClient(model=config.model, temperature=config.temperature, seed=config.seed),
            executor=MCPToolExecutor(server, timeout=config.tool_timeout),
            trace=TraceLog.in_var_dir(context.var_dir),
            config=config,
            spend=SpendGuard.for_model(config.model, cap),
            context={"retrieval": retrieval.describe(), "spend_cap_usd": cap},
        )
