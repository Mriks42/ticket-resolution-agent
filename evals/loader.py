"""Load and cross-validate the eval dataset."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evals.schema import Customer, Ticket, Tool

DATASET_DIR = Path(__file__).parent / "dataset"
POLICY_DIR = DATASET_DIR / "policies"
TICKETS_PATH = DATASET_DIR / "tickets.json"
CUSTOMERS_PATH = DATASET_DIR / "customers.json"

_FRONTMATTER_ID = re.compile(r"^policy_id:\s*(\S+)\s*$", re.MULTILINE)


def load_policy_ids(policy_dir: Path = POLICY_DIR) -> set[str]:
    """Policy IDs declared in the corpus, checked against their filenames."""
    ids: set[str] = set()
    for path in sorted(policy_dir.glob("POL-*.md")):
        match = _FRONTMATTER_ID.search(path.read_text(encoding="utf-8"))
        if not match:
            raise ValueError(f"{path.name}: no policy_id in frontmatter")
        declared = match.group(1)
        if declared != path.stem:
            raise ValueError(
                f"{path.name}: declares policy_id {declared!r}, filename says {path.stem!r}"
            )
        ids.add(declared)
    if not ids:
        raise ValueError(f"no policy documents found in {policy_dir}")
    return ids


def load_customers(customers_path: Path = CUSTOMERS_PATH) -> dict[str, Customer]:
    """Account fixtures backing `lookup_customer` and `search_past_tickets`."""
    raw = json.loads(customers_path.read_text(encoding="utf-8"))
    customers = {cid: Customer.model_validate(rec) for cid, rec in raw.items()}
    mismatched = [cid for cid, c in customers.items() if c.customer_id != cid]
    if mismatched:
        raise ValueError(f"customer records keyed inconsistently: {mismatched}")
    return customers


def load_tickets(
    tickets_path: Path = TICKETS_PATH,
    policy_dir: Path = POLICY_DIR,
    customers_path: Path = CUSTOMERS_PATH,
) -> list[Ticket]:
    """Parse tickets and fail loudly on any label the fixtures cannot support."""
    raw = json.loads(tickets_path.read_text(encoding="utf-8"))
    tickets = [Ticket.model_validate(item) for item in raw]

    duplicates = [tid for tid, n in Counter(t.ticket_id for t in tickets).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate ticket ids: {sorted(duplicates)}")

    known = load_policy_ids(policy_dir)
    dangling = {
        t.ticket_id: sorted(set(t.labels.relevant_policy_ids) - known)
        for t in tickets
        if set(t.labels.relevant_policy_ids) - known
    }
    if dangling:
        raise ValueError(f"tickets cite policy ids not in the corpus: {dangling}")

    customers = load_customers(customers_path)
    unknown = sorted({t.customer_id for t in tickets} - customers.keys())
    if unknown:
        raise ValueError(f"tickets reference customers with no fixture: {unknown}")

    # A tool only belongs in expected_tools if it can actually inform the
    # answer. search_past_tickets against a customer with no history returns
    # empty, and crediting that call would inflate tool-call precision.
    barren = [
        t.ticket_id
        for t in tickets
        if Tool.SEARCH_PAST_TICKETS in t.labels.expected_tools
        and not customers[t.customer_id].past_tickets
    ]
    if barren:
        raise ValueError(
            f"tickets expect search_past_tickets but their customer has no "
            f"prior history: {barren}"
        )

    return tickets
