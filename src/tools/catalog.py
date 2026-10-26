"""The six tools, their permission scopes, and their JSON schemas.

The catalog is declarative: each tool states the scope it requires and whether
it is gated. It does not check anything. Enforcement lives in
`src.mcp.server`, which is the only component that knows what scopes the
caller was granted — a tool cannot be tricked into approving itself because it
is never told who is calling.

Scopes are per-tool and read scopes never overlap write scopes, so granting an
agent the ability to read the policy corpus cannot transitively grant it the
ability to move money.

**Gated tools are marked, not gated.** `issue_refund`, `send_reply` and
`escalate` carry `gated=True` and their results carry `"gated": true`, but no
approval is checked yet — the HITL gate and the PENDING_APPROVAL state are a
later session. Right now a caller holding the write scope executes the write.
The marking is the seam that gate will attach to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.tools.context import ToolContext
from src.tools.errors import ToolInputError

#: Chosen from the recall sweep in `evals/retrieval_baseline.py`, not picked as
#: a round number. Recall rises in two steps, at k=8 and k=10, then goes flat:
#: 0.716 at k=10 against 0.727 at both k=12 and k=15. 10 is the knee — the last
#: k that buys anything before diminishing returns. See ADR 0003.
DEFAULT_RETRIEVAL_LIMIT = 10


class Scope(str, Enum):
    """Permission scopes. One per tool; no tool shares a scope with another."""

    CUSTOMER_READ = "customer:read"
    TICKETS_READ = "tickets:read"
    POLICY_READ = "policy:read"
    REFUND_WRITE = "refund:write"
    REPLY_WRITE = "reply:write"
    ESCALATION_WRITE = "escalation:write"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    scope: Scope
    gated: bool
    input_schema: dict[str, Any]
    handler: Callable[..., dict]


# --- read-only handlers ----------------------------------------------------


def lookup_customer(ctx: ToolContext, *, customer_id: str) -> dict:
    """The account, then its history from memory, oldest first."""
    record = ctx.fixtures.customer(customer_id)
    record["past_tickets"] = [
        {"ticket_id": e.ticket_id, "opened": e.opened, "summary": e.summary, "resolution": e.resolution}
        for e in ctx.memory.episodes.history(customer_id)
    ]
    return record


def search_past_tickets(
    ctx: ToolContext,
    *,
    query: str,
    customer_id: str | None = None,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> dict:
    """Ranked by similarity to `query`. Scoped to one customer, it returns that
    customer's whole history up to `limit`: empty means no history (ADR 0008)."""
    if customer_id is not None:
        ctx.fixtures.check_customer(customer_id)
    hits = ctx.memory.episodes.search(query, customer_id, _limit(limit))
    matches = [
        {
            "ticket_id": e.ticket_id,
            "customer_id": e.customer_id,
            "opened": e.opened,
            "summary": e.summary,
            "resolution": e.resolution,
            "score": round(score, 6),
        }
        for e, score in hits
    ]
    return {"query": query, "customer_id": customer_id, "matches": matches}


def retrieve_policy(
    ctx: ToolContext,
    *,
    query: str,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
) -> dict:
    """Clause-level passages. Empty when nothing in the corpus matches."""
    hits = ctx.fixtures.policy.search(query, _limit(limit))
    return {
        "query": query,
        "passages": [
            {**clause.as_dict(), "score": round(score, 6)} for clause, score in hits
        ],
    }


# --- write handlers --------------------------------------------------------


def issue_refund(
    ctx: ToolContext, *, ticket_id: str, amount: float, reason: str
) -> dict:
    _require_text("ticket_id", ticket_id)
    _require_text("reason", reason)
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ToolInputError(f"amount must be a number, got {type(amount).__name__}")
    if amount <= 0:
        raise ToolInputError(f"amount must be greater than zero, got {amount}")

    record = ctx.store.append(
        "refunds",
        {
            "ticket_id": ticket_id,
            "amount": float(amount),
            "currency": "USD",
            "reason": reason,
            "gated": True,
        },
    )
    return {"refund_id": record["id"], "status": "recorded", "gated": True, **record}


def send_reply(ctx: ToolContext, *, ticket_id: str, body: str) -> dict:
    _require_text("ticket_id", ticket_id)
    _require_text("body", body)

    record = ctx.store.append(
        "replies", {"ticket_id": ticket_id, "body": body, "gated": True}
    )
    return {"reply_id": record["id"], "status": "recorded", "gated": True, **record}


def escalate(
    ctx: ToolContext, *, ticket_id: str, reason: str, trace: list | None = None
) -> dict:
    _require_text("ticket_id", ticket_id)
    _require_text("reason", reason)
    if trace is not None and not isinstance(trace, list):
        raise ToolInputError(f"trace must be a list, got {type(trace).__name__}")

    record = ctx.store.append(
        "escalations",
        {
            "ticket_id": ticket_id,
            "reason": reason,
            "trace": trace if trace is not None else [],
            "gated": True,
        },
    )
    return {
        "escalation_id": record["id"],
        "status": "recorded",
        "gated": True,
        **record,
    }


def _require_text(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{field} must be a non-empty string")


def _limit(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ToolInputError(f"limit must be a positive integer, got {value!r}")
    return min(value, 50)


# --- schemas ---------------------------------------------------------------

_QUERY = {"type": "string", "description": "Natural-language search text."}
_LIMIT = {
    "type": "integer",
    "minimum": 1,
    "maximum": 50,
    "default": DEFAULT_RETRIEVAL_LIMIT,
}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


CATALOG: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="lookup_customer",
            description=(
                "Fetch the account record for a customer: plan, seats, billing "
                "cycle, account status, last charge, and the verified owner "
                "email."
            ),
            scope=Scope.CUSTOMER_READ,
            gated=False,
            input_schema=_schema(
                {"customer_id": {"type": "string", "description": "e.g. CUST-1042"}},
                ["customer_id"],
            ),
            handler=lookup_customer,
        ),
        ToolSpec(
            name="search_past_tickets",
            description=(
                "Search this customer's prior contacts and how they were "
                "resolved. Returns an empty list when the customer has no "
                "history; that is an answer, not a failure."
            ),
            scope=Scope.TICKETS_READ,
            gated=False,
            input_schema=_schema(
                {
                    "query": _QUERY,
                    "customer_id": {
                        "type": ["string", "null"],
                        "description": "Restrict to one customer. Omit to search all.",
                    },
                    "limit": _LIMIT,
                },
                ["query"],
            ),
            handler=search_past_tickets,
        ),
        ToolSpec(
            name="retrieve_policy",
            description=(
                "Retrieve policy passages relevant to a question. Each passage "
                "carries a clause-level id (e.g. REF-001.3) which is what a "
                "reply must cite. Returns an empty list when no clause matches; "
                "do not answer from memory in that case."
            ),
            scope=Scope.POLICY_READ,
            gated=False,
            input_schema=_schema({"query": _QUERY, "limit": _LIMIT}, ["query"]),
            handler=retrieve_policy,
        ),
        ToolSpec(
            name="issue_refund",
            description=(
                "Issue a refund against a ticket. Gated: subject to approval "
                "thresholds in POL-ESC-001."
            ),
            scope=Scope.REFUND_WRITE,
            gated=True,
            input_schema=_schema(
                {
                    "ticket_id": {"type": "string"},
                    "amount": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "USD.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Cite the clause that permits it.",
                    },
                },
                ["ticket_id", "amount", "reason"],
            ),
            handler=issue_refund,
        ),
        ToolSpec(
            name="send_reply",
            description=(
                "Send a reply to the customer. Gated: a reply making a policy "
                "claim requires that claim to be backed by a retrieved passage."
            ),
            scope=Scope.REPLY_WRITE,
            gated=True,
            input_schema=_schema(
                {"ticket_id": {"type": "string"}, "body": {"type": "string"}},
                ["ticket_id", "body"],
            ),
            handler=send_reply,
        ),
        ToolSpec(
            name="escalate",
            description=(
                "Queue the ticket for a human, with the reasoning trace "
                "attached. Use when policy is silent, two clauses conflict, or "
                "POL-ESC-001.4 applies."
            ),
            scope=Scope.ESCALATION_WRITE,
            gated=True,
            input_schema=_schema(
                {
                    "ticket_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "trace": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Ordered state transitions and observations.",
                    },
                },
                ["ticket_id", "reason"],
            ),
            handler=escalate,
        ),
    )
}
