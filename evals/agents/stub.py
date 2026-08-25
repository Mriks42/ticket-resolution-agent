"""Floor baseline: escalates every ticket, calls nothing else."""

from __future__ import annotations

from uuid import uuid4

from evals.schema import AgentResult, Outcome, Ticket, Tool, ToolCall


class AlwaysEscalateAgent:
    """Resolves nothing. Its scores are the floor a real agent must beat."""

    name = "always-escalate"

    async def run(self, ticket: Ticket) -> AgentResult:
        return AgentResult(
            ticket_id=ticket.ticket_id,
            run_id=f"stub-{uuid4().hex[:12]}",
            outcome=Outcome.ESCALATE,
            tool_calls=[
                ToolCall(
                    tool=Tool.ESCALATE,
                    args={
                        "ticket_id": ticket.ticket_id,
                        "reason": "stub agent escalates unconditionally",
                        "trace": [],
                    },
                )
            ],
            cited_policy_ids=[],
        )
