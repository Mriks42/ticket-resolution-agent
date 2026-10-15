"""The read side of the tools, backed by `evals/dataset/`.

This module is the **only** place in `src/` that reaches into `evals/`. That is
deliberate. The account data the agent reads at runtime will eventually come
from Postgres, and when it does, this file is the one that changes — the
catalog, the server, and the agent loop all sit behind it.

The direction of the dependency is worth naming: `src.tools` importing
`evals.loader` is backwards from where this ends up. It is here because
CLAUDE.md forbids mocked results in the eval path, and the fixtures are the
only real customer data that exists. Reusing `evals.loader` rather than
re-reading the JSON means the same validation guards a tool call and a dataset
load, so the two cannot drift apart.

Nothing here is generated at call time. Every value a read tool returns is
already sitting in `evals/dataset/`.
"""

from __future__ import annotations

from pathlib import Path

from evals.loader import load_customers
from src.paths import DEFAULT_DATASET_DIR
from src.tools.errors import ToolInputError
from src.tools.lexical import LexicalIndex
from src.tools.retrieval import PolicySearch, RetrievalConfig, build_policy_index


class FixtureBackend:
    """Read-only access to the customer records and the policy corpus.

    `retrieval` picks the policy backend; `None` reads it from the environment.
    """

    def __init__(
        self,
        dataset_dir: Path | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir or DEFAULT_DATASET_DIR)
        self._customers = load_customers(self.dataset_dir / "customers.json")
        self._policy = build_policy_index(self.dataset_dir / "policies", retrieval)

        # Past contacts are indexed once, keyed "CUST-1520/H-4471" so a search
        # can be filtered by customer without rebuilding the index.
        self._past_index = LexicalIndex(
            {
                f"{customer_id}/{past.ticket_id}": f"{past.summary} {past.resolution}"
                for customer_id, customer in self._customers.items()
                for past in customer.past_tickets
            }
        )

    @property
    def policy(self) -> PolicySearch:
        return self._policy

    def customer(self, customer_id: str) -> dict:
        """The account record `lookup_customer` returns."""
        record = self._customers.get(customer_id)
        if record is None:
            raise ToolInputError(f"no customer record for {customer_id!r}")
        return record.model_dump(mode="json")

    def search_past_tickets(
        self,
        query: str,
        customer_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Prior contacts matching `query`, optionally for one customer only.

        An unknown `customer_id` raises; a *known* customer with no history
        returns an empty list, because "this customer has never written in"
        is an answer the agent should be able to act on.
        """
        if customer_id is not None and customer_id not in self._customers:
            raise ToolInputError(f"no customer record for {customer_id!r}")

        matches: list[dict] = []
        for hit in self._past_index.search(query):
            owner, _, ticket_id = hit.doc_id.partition("/")
            if customer_id is not None and owner != customer_id:
                continue
            past = next(
                p
                for p in self._customers[owner].past_tickets
                if p.ticket_id == ticket_id
            )
            matches.append(
                {
                    "ticket_id": past.ticket_id,
                    "customer_id": owner,
                    "opened": past.opened,
                    "summary": past.summary,
                    "resolution": past.resolution,
                    "score": round(hit.score, 6),
                }
            )
            if len(matches) >= limit:
                break
        return matches
