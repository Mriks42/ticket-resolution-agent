"""Dataset and agent-interface types for the eval harness."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

POLICY_ID_RE = re.compile(r"^POL-[A-Z]{3}-\d{3}$")


class Outcome(str, Enum):
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    PENDING_APPROVAL = "pending_approval"


class Tool(str, Enum):
    LOOKUP_CUSTOMER = "lookup_customer"
    SEARCH_PAST_TICKETS = "search_past_tickets"
    RETRIEVE_POLICY = "retrieve_policy"
    ISSUE_REFUND = "issue_refund"
    SEND_REPLY = "send_reply"
    ESCALATE = "escalate"


class Confidence(str, Enum):
    """How sure the label author was. Anything below HIGH needs review."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    expected_tools: list[Tool]
    relevant_policy_ids: list[str]
    confidence: Confidence = Confidence.HIGH
    label_note: str | None = None

    @field_validator("relevant_policy_ids")
    @classmethod
    def _well_formed_policy_ids(cls, ids: list[str]) -> list[str]:
        bad = [i for i in ids if not POLICY_ID_RE.match(i)]
        if bad:
            raise ValueError(f"malformed policy ids: {bad}")
        return ids

    @model_validator(mode="after")
    def _uncertain_labels_state_why(self) -> TicketLabels:
        if self.confidence is not Confidence.HIGH and not self.label_note:
            raise ValueError(
                f"confidence={self.confidence.value} requires a label_note"
            )
        return self


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    customer_id: str
    requester_email: str
    body: str
    labels: TicketLabels


class Plan(str, Enum):
    FREE = "free"
    PRO = "pro"
    TEAM = "team"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    DUNNING = "dunning"
    GRACE = "grace"
    SUSPENDED = "suspended"


class PastTicket(BaseModel):
    """A prior contact from this customer, for `search_past_tickets`."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    opened: str
    summary: str
    resolution: str


class Customer(BaseModel):
    """The account state `lookup_customer` returns during an eval run."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    owner_email: str
    plan: Plan
    seats: int
    billing_cycle: Literal["monthly", "annual"] | None
    account_status: AccountStatus
    last_charge_usd: float | None
    last_charge_date: str | None
    past_tickets: list[PastTicket] = []


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Tool
    args: dict[str, Any] = {}


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    run_id: str
    outcome: Outcome
    tool_calls: list[ToolCall] = []
    cited_policy_ids: list[str] = []
    #: The loop's terminal state. Distinct from `outcome` in one case: FAILED
    #: (no usable completion from the model) is reported as an escalation,
    #: since the ticket still has to reach a human, but must stay countable.
    final_state: str | None = None
    #: Why the run stopped, in words. Set when the loop, not the model, ended it.
    stop_reason: str | None = None
    #: Hallucinated-policy events in the drafted reply, "kind: detail", as the
    #: guardrail found them — whether or not it blocked the reply (ADR 0007).
    policy_violations: list[str] = []
    #: What the run paid, LLM and embeddings, in USD. None when the agent has
    #: no price on record: unknown, not free (ADR 0009).
    cost_usd: float | None = None
    #: Wall clock from the run's start to its terminal state.
    elapsed_s: float | None = None


@runtime_checkable
class Agent(Protocol):
    """The interface `evals/run.py` drives. The real agent implements this."""

    name: str

    async def run(self, ticket: Ticket) -> AgentResult: ...
