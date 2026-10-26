"""The service behind the API: runs tickets in the background, one at a time,
and carries out what a human approves.

**One run at a time, in the order submitted.** Every run takes the same lock.
That keeps per-run embedding cost exact (ADR 0009), and keeps memory and
billing writes in submission order, which is how an eval run sees them. It
also caps throughput at one ticket per agent run; a deployment that needs
more runs more instances.

**The trace is the record of a run.** A run's status lives in memory while
the process does, and its events are read from the trace, so a finished run
can still be looked up after a restart. A run queued but not started when the
process stops is lost; one started but not finished shows as interrupted.

**An approval executes what the gate held, through the write-scoped server.**
Refunds first, then the held reply, each audited under the original run id,
with no retries (ADR 0006). An executed refund joins the account's billing
history (ADR 0011). The first action that does not succeed stops the rest,
so a reply never tells a customer a refund went through when it did not.
Memory is not rewritten: the close recorded the refund as held, which was true
when it was written.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import date

from evals.schema import AgentResult, Outcome
from src.agent.loop import AgentLoop
from src.api.approvals import ApprovalLog
from src.api.models import ApprovalDecision, IncomingTicket
from src.tools.context import ToolContext


class ApprovalConflict(Exception):
    """The approval has already been decided."""


class AgentService:
    def __init__(
        self,
        *,
        agent: AgentLoop,
        context: ToolContext,
        approvals: ApprovalLog | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._agent = agent
        self._context = context
        self._approvals = approvals or ApprovalLog.in_var_dir(context.var_dir)
        self._today = today
        self._lock = asyncio.Lock()
        self._runs: dict[str, dict] = {}
        self._tasks: set[asyncio.Task] = set()

    # --- tickets and runs ----------------------------------------------------

    def submit(self, ticket: IncomingTicket) -> str:
        """Queue a run and return its id. An unknown customer raises
        `ToolInputError` before anything is queued."""
        self._context.fixtures.check_customer(ticket.customer_id)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._runs[run_id] = {"run_id": run_id, "ticket_id": ticket.ticket_id, "status": "queued"}
        task = asyncio.create_task(self._process(run_id, ticket))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run_id

    async def _process(self, run_id: str, ticket: IncomingTicket) -> None:
        async with self._lock:
            state = self._runs[run_id]
            state["status"] = "running"
            try:
                result = await self._agent.run(ticket, run_id=run_id)
            except Exception as exc:  # noqa: BLE001 - e.g. the spend cap; recorded, not swallowed
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                return
            state.update(status="finished", result=_result_view(result))
            if result.outcome is Outcome.PENDING_APPROVAL and result.held:
                approval = self._approvals.open(
                    run_id=run_id, ticket_id=ticket.ticket_id,
                    customer_id=ticket.customer_id, held=result.held,
                )
                state["result"]["approval_id"] = approval["approval_id"]

    async def idle(self) -> None:
        """Wait for every queued run to finish."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks))

    def run(self, run_id: str) -> dict | None:
        """The run's status and result, and its trace events. None if this
        process never saw it and the trace has no record of it."""
        events = self._agent.trace.records(run_id)
        known = self._runs.get(run_id)
        if known is None and not events:
            return None
        if known is None:
            finished = [e for e in events if e["event"] == "run_finished"]
            known = {"run_id": run_id, "status": "finished" if finished else "interrupted"}
            if finished:
                known["result"] = {
                    k: finished[-1].get(k) for k in ("outcome", "final_state", "stop_reason", "elapsed_s")
                }
        return {**known, "events": events}

    # --- approvals -----------------------------------------------------------------

    def pending_approvals(self) -> list[dict]:
        return self._approvals.pending()

    async def decide(self, approval_id: str, decision: ApprovalDecision) -> dict:
        """Carry out or discard what was held. Raises `KeyError` for an unknown
        approval and `ApprovalConflict` for one already decided."""
        async with self._lock:
            approval = self._approvals.get(approval_id)
            if approval is None:
                raise KeyError(approval_id)
            if approval["status"] != "pending":
                raise ApprovalConflict(f"{approval_id} was already {approval['status']}")

            executed: list[dict] = []
            status = "rejected"
            if decision.decision == "approve":
                status = "approved"
                for action in approval["held"]:
                    obs = await self._agent.writer.call(
                        action["tool"], action["args"], run_id=approval["run_id"]
                    )
                    executed.append({"tool": action["tool"], "status": obs.status, "message": obs.message})
                    if obs.status != "ok":
                        status = "failed"
                        break
                    if action["tool"] == "issue_refund":
                        self._context.billing.record_refund(
                            approval["customer_id"], amount_usd=float(action["args"]["amount"]),
                            date=self._today().isoformat(), ticket_id=approval["ticket_id"],
                        )
            self._agent.trace.emit(
                approval["run_id"], "approval_decided",
                approval_id=approval_id, status=status, decided_by=decision.decided_by,
                note=decision.note, executed=executed,
            )
            return self._approvals.decide(
                approval_id, status=status, decided_by=decision.decided_by,
                note=decision.note, executed=executed,
            )


def _result_view(result: AgentResult) -> dict:
    return {
        "outcome": result.outcome.value,
        "final_state": result.final_state,
        "stop_reason": result.stop_reason,
        "tool_calls": [c.tool.value for c in result.tool_calls],
        "cited_policy_ids": result.cited_policy_ids,
        "policy_violations": result.policy_violations,
        "cost_usd": result.cost_usd,
        "elapsed_s": result.elapsed_s,
    }
