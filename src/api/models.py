"""Request bodies. Responses are plain dicts, documented on the routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IncomingTicket(BaseModel):
    """A ticket as a caller submits it: the four fields the agent is shown,
    and nothing that could carry an answer."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=64)
    requester_email: str = Field(min_length=3, max_length=320)
    body: str = Field(min_length=1, max_length=20_000)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    #: Who decided, for the record. The approver key proves the caller may
    #: approve; it does not say who they are.
    decided_by: str = Field(min_length=1, max_length=200)
    note: str | None = Field(default=None, max_length=2_000)
