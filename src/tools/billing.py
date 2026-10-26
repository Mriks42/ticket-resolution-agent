"""Billing history: the refunds each customer has received.

Account state, not memory (ADR 0011). The gate counts it for REF-002.4 ("two
or more refunds in the preceding 12 months requires approval for any further
refund"), and `lookup_customer` shows it, whatever the memory switch says: a
money control must not change with an ablation of what the agent remembers.

It starts from the fixtures' `refunds`, and every refund the gate executes is
added, so a later ticket from the same customer in the same run counts it.
A held refund is not added: the customer has not received it. One instance
per `ToolContext`, like memory, so one eval run's refunds never reach the
next.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from evals.schema import Customer


class BillingHistory:
    def __init__(self, seed: Mapping[str, list[dict]] | None = None) -> None:
        self._refunds: dict[str, list[dict]] = defaultdict(list)
        for customer_id, records in (seed or {}).items():
            self._refunds[customer_id].extend(dict(r) for r in records)

    @classmethod
    def from_customers(cls, customers: Mapping[str, Customer]) -> BillingHistory:
        return cls({
            cid: [r.model_dump(mode="json") for r in c.refunds] for cid, c in customers.items()
        })

    def refunds(self, customer_id: str) -> list[dict]:
        """Oldest first, as recorded. A copy: callers cannot edit history."""
        return [dict(r) for r in self._refunds.get(customer_id, [])]

    def record_refund(
        self, customer_id: str, *, amount_usd: float, date: str, ticket_id: str | None
    ) -> None:
        self._refunds[customer_id].append(
            {"date": date, "amount_usd": float(amount_usd), "ticket_id": ticket_id}
        )
