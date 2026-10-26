"""What a finished run hands to memory.

The loop builds a `Close` from its own records: the terminal state, the
refunds the gate executed or held, the clause ids the run cited, and the
ticket body. Nothing in it is model prose. Memory is read back as the
customer's own records, and the guardrail grounds replies in those, so a
sentence the model wrote in one run must not be able to ground a claim in the
next (ADR 0008).
"""

from __future__ import annotations

from dataclasses import dataclass

REFUND_STATUSES = ("issued", "held")


@dataclass(frozen=True)
class Refund:
    amount: float
    #: "issued": the gate executed it. "held": waiting for an approver.
    status: str

    def __post_init__(self) -> None:
        if self.status not in REFUND_STATUSES:
            raise ValueError(f"refund status must be one of {REFUND_STATUSES}, got {self.status!r}")


@dataclass(frozen=True)
class Close:
    run_id: str
    customer_id: str
    ticket_id: str
    #: The customer's own words, already grounding in the run that wrote them.
    body: str
    #: RESOLVE, ESCALATE, PENDING_APPROVAL or FAILED.
    final_state: str
    #: ISO date of the close.
    date: str
    refunds: tuple[Refund, ...] = ()
    #: Clause ids the run cited or rested on, sorted. Ids only, never text.
    clauses: tuple[str, ...] = ()
