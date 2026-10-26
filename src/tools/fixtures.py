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

Nothing here is generated at call time. Every value this module returns is
already sitting in `evals/dataset/`. History the agent writes during a run is
memory's (`src/memory/`), seeded from here.
"""

from __future__ import annotations

from pathlib import Path

from evals.loader import load_customers
from src.memory.episodic import Episode
from src.paths import DEFAULT_DATASET_DIR
from src.tools.errors import ToolInputError
from src.tools.retrieval import PolicySearch, RetrievalConfig, build_policy_index


class FixtureBackend:
    """Read-only access to the customer records and the policy corpus.

    `retrieval` picks the policy backend; `None` reads it from the environment.
    A customer's history is not read from here at call time: it seeds memory
    (`history_seed`), and memory answers for it from then on (ADR 0008).
    """

    def __init__(
        self,
        dataset_dir: Path | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir or DEFAULT_DATASET_DIR)
        self._customers = load_customers(self.dataset_dir / "customers.json")
        self._policy = build_policy_index(self.dataset_dir / "policies", retrieval)

    @property
    def policy(self) -> PolicySearch:
        return self._policy

    def customer(self, customer_id: str) -> dict:
        """The account record `lookup_customer` returns, without its history."""
        self.check_customer(customer_id)
        return self._customers[customer_id].model_dump(mode="json", exclude={"past_tickets"})

    def check_customer(self, customer_id: str) -> None:
        if customer_id not in self._customers:
            raise ToolInputError(f"no customer record for {customer_id!r}")

    def history_seed(self) -> list[Episode]:
        """Every customer's `past_tickets`: the prior history memory starts from."""
        return [
            Episode(customer_id, p.ticket_id, p.opened, p.summary, p.resolution)
            for customer_id, customer in self._customers.items()
            for p in customer.past_tickets
        ]
