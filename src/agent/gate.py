"""The HITL gate: deterministic rules for gated actions.

CLAUDE.md requires approval to rest on deterministic rules — action type
combined with retrieval coverage — and never on model self-rated confidence or
logprobs. Every function here is pure: given an action and facts the gate holds,
it returns a `Decision`. The loop supplies the facts and carries the decision
out.

Three verdicts:

- `EXECUTE`   the action runs, through a write-scoped server only the gate
              reaches
- `HOLD`      the action waits for a named approver; the ticket ends in
              PENDING_APPROVAL unless the model escalates
- `ESCALATE`  the action must not run and a human takes the ticket

The gate is the *action-type* half of CLAUDE.md's rule. The *coverage* half —
whether a claim is backed by a retrieved passage — is the guardrail
(`src/agent/guardrail.py`), which the loop runs first. It moved there from
here in ADR 0007, so the "without guardrail" ablation can switch every
claim-against-retrieval check off together.

The rules, in the order they are applied:

**Refunds.** (1) Owner check, ESC-001.4: a refund for anyone other than the
verified account owner escalates, whatever the amount. An owner the gate cannot
verify fails closed. (2) Band, ESC-001.1–.3, applied to the ticket's running
total (refunds already executed or held, plus this one): up to USD 50
executes, up to USD 500 waits for a support manager, above that for finance.
Banding the total rather than each refund means a refund cannot be split into
pieces that each look self-serve; ADR 0006 records the decision. (3) Refund
history, REF-002.4: a customer who has received two or more refunds in the
preceding 12 months needs approval for any further refund, so a refund the
band would have executed waits for a support manager instead. It never
lowers a band. The count comes from the account's billing history, which is
account state rather than memory (ADR 0011).

**Replies.** If an approval is pending, the reply is held with it as one
package; otherwise it executes. The owner check does not apply to replies
(decided with the label owner): a non-owner can be answered, not refunded.

The band thresholds are policy copied into code, which ADR 0003 decided on
purpose: the bands are arithmetic on a number the agent holds, not something
retrieval can be relied on to surface. `tests/test_gate.py` checks them
against the clause text, so the copy cannot drift silently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

#: ESC-001.1: "Support may issue a refund of up to USD 50 without approval."
SELF_SERVE_REFUND_MAX = Decimal("50.00")
#: ESC-001.2: "between USD 50.01 and USD 500 requires support manager approval".
#: Above it, ESC-001.3: "requires finance approval".
MANAGER_REFUND_MAX = Decimal("500.00")
#: REF-002.4: "A customer who has received two or more refunds in the
#: preceding 12 months requires approval for any further refund".
REFUND_HISTORY_LIMIT = 2
REFUND_HISTORY_DAYS = 365

class Verdict(str, Enum):
    EXECUTE = "execute"
    HOLD = "hold"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    rule: str
    detail: str
    clause: str | None = None
    approver: str | None = None


def decide_refund(
    *,
    amount: float,
    already_refunded: float,
    requester_email: str,
    owner_email: str | None,
    prior_refunds: int = 0,
) -> Decision:
    if owner_email is None:
        return Decision(
            Verdict.ESCALATE, "owner_unverified",
            "the account owner could not be looked up, so the requester cannot be verified",
            clause="ESC-001.4",
        )
    if _normal(requester_email) != _normal(owner_email):
        return Decision(
            Verdict.ESCALATE, "non_owner",
            "refund requested by someone other than the verified account owner",
            clause="ESC-001.4",
        )

    cents = _cents(amount)
    total = _cents(already_refunded) + cents
    on_ticket = f"USD {cents}, bringing this ticket's refunds to USD {total},"
    if total <= SELF_SERVE_REFUND_MAX and prior_refunds >= REFUND_HISTORY_LIMIT:
        return Decision(
            Verdict.HOLD, "refund_history",
            f"{on_ticket} is self-serve by amount, but the customer has received "
            f"{prior_refunds} refunds in the preceding 12 months, so it needs approval",
            clause="REF-002.4", approver="support manager",
        )
    if total <= SELF_SERVE_REFUND_MAX:
        return Decision(
            Verdict.EXECUTE, "refund_band",
            f"{on_ticket} is within the USD {SELF_SERVE_REFUND_MAX} self-serve limit",
            clause="ESC-001.1",
        )
    if total <= MANAGER_REFUND_MAX:
        return Decision(
            Verdict.HOLD, "refund_band",
            f"{on_ticket} needs support manager approval",
            clause="ESC-001.2", approver="support manager",
        )
    return Decision(
        Verdict.HOLD, "refund_band",
        f"{on_ticket} is above USD {MANAGER_REFUND_MAX} and needs finance approval",
        clause="ESC-001.3", approver="finance",
    )


def recent_refunds(history: Iterable[Mapping], today: date) -> int:
    """Refunds in `history` received in the 12 months before `today`, today
    included. A record without a readable date counts: failing closed holds a
    refund rather than letting one through."""
    since = today - timedelta(days=REFUND_HISTORY_DAYS)
    count = 0
    for record in history:
        try:
            received = date.fromisoformat(str(record.get("date")))
        except ValueError:
            count += 1
            continue
        if since < received <= today:
            count += 1
    return count


def decide_reply(*, pending: bool) -> Decision:
    if pending:
        return Decision(
            Verdict.HOLD, "approval_pending",
            "held with the pending approval, so the customer is not told before it is decided",
        )
    return Decision(Verdict.EXECUTE, "nothing_pending", "no approval is pending on this ticket")


def _cents(amount: float) -> Decimal:
    return Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _normal(email: str) -> str:
    return email.strip().lower()
