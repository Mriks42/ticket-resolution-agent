"""Semantic memory: durable facts about a customer, folded in at each close.

One record per customer, built by `fold` — a pure function of the record so
far and one `Close`. There is no LLM in it: a fact the model invented would be
read back in a later ticket as the customer's own record (ADR 0008). The
facts are:

- `tickets_closed`, counting closes, so a ticket handled twice counts twice;
- `refunds`, each with ticket, amount, status (`issued` or `held`) and date;
- `escalations`, each with ticket, date and the clause ids the run rested on.
  A FAILED run is an escalation: the ticket reached a human either way;
- `policies_cited`, how many closed tickets cited each policy document, and
  `recurring`, the documents cited at two or more — a deterministic stand-in
  for "recurring issue". POL-ESC-001 is never recurring: it governs
  procedure, not what the ticket was about.

Plan tier is deliberately absent. The account record is the authority on it,
and a remembered copy can only go stale.

Two stores, one contract, picked like the episodic ones: a dict in-process,
or a Postgres row per namespace and customer. The column is `json`, not
`jsonb`, because `jsonb` reorders keys and the key order reaches the model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Protocol

import psycopg
from psycopg.types.json import Json

from src.memory.close import Close
from src.tools.errors import ToolError

#: Escalation procedure, cited on every escalation and every refund band.
NEVER_RECURRING = frozenset({"POL-ESC-001"})
ESCALATED = frozenset({"ESCALATE", "FAILED"})


def policy_id(clause_id: str) -> str:
    """REF-001.3 -> POL-REF-001."""
    return f"POL-{clause_id.split('.')[0]}"


def fold(facts: Mapping[str, Any], close: Close) -> dict:
    """The customer's facts after `close`. Pure: the same closes in the same
    order always give the same facts, and `facts` is not modified."""
    cited = Counter(facts.get("policies_cited", {}))
    cited.update({policy_id(c) for c in close.clauses})
    escalation = (
        [{"ticket_id": close.ticket_id, "date": close.date, "clauses": list(close.clauses)}]
        if close.final_state in ESCALATED
        else []
    )
    return {
        "tickets_closed": facts.get("tickets_closed", 0) + 1,
        "refunds": [
            *facts.get("refunds", []),
            *(
                {"ticket_id": close.ticket_id, "amount": r.amount, "status": r.status, "date": close.date}
                for r in close.refunds
            ),
        ],
        "escalations": [*facts.get("escalations", []), *escalation],
        "policies_cited": dict(sorted(cited.items())),
        "recurring": sorted(p for p, n in cited.items() if n >= 2 and p not in NEVER_RECURRING),
    }


class FactStoreError(ToolError):
    """The fact store could not answer or could not write."""


class FactStore(Protocol):
    backend: str
    namespace: str

    def get(self, customer_id: str) -> dict: ...

    def put(self, customer_id: str, facts: dict) -> None: ...


class InProcessFacts:
    backend = "in-process"

    def __init__(self, *, namespace: str) -> None:
        self.namespace = namespace
        self._facts: dict[str, dict] = {}

    def get(self, customer_id: str) -> dict:
        return self._facts.get(customer_id, {})

    def put(self, customer_id: str, facts: dict) -> None:
        self._facts[customer_id] = facts


SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_facts (
    namespace    text        NOT NULL,
    customer_id  text        NOT NULL,
    facts        json        NOT NULL,
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, customer_id)
);
"""


class PgFacts:
    backend = "postgres"

    def __init__(self, *, conn: psycopg.Connection, namespace: str) -> None:
        self.namespace = namespace
        self._conn = conn
        conn.execute(SCHEMA)

    def get(self, customer_id: str) -> dict:
        try:
            row = self._conn.execute(
                "SELECT facts FROM customer_facts WHERE namespace = %s AND customer_id = %s",
                (self.namespace, customer_id),
            ).fetchone()
        except psycopg.Error as exc:
            raise FactStoreError(f"fact read failed: {exc}") from exc
        return row[0] if row else {}

    def put(self, customer_id: str, facts: dict) -> None:
        try:
            self._conn.execute(
                "INSERT INTO customer_facts (namespace, customer_id, facts) VALUES (%s, %s, %s) "
                "ON CONFLICT (namespace, customer_id) DO UPDATE "
                "SET facts = EXCLUDED.facts, updated_at = now()",
                (self.namespace, customer_id, Json(facts)),
            )
        except psycopg.Error as exc:
            raise FactStoreError(f"fact write failed: {exc}") from exc
