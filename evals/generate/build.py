"""Build one batch of synthetic tickets for label review.

    uv run python -m evals.generate.build --batch 1

Writes `evals/dataset/synthetic/batch_NN.json` with `"reviewed": false`, and
adds the batch's customers to `evals/dataset/synthetic/customers.json`. The
batch stays out of every eval run until the label owner reviews it and sets
`"reviewed": true` (CLAUDE.md: labels are never auto-accepted).

What is generated and what is not:

- **Hand-written:** each spec in `scenarios.py` — the account, who is asking,
  the facts, the label and the clauses behind it.
- **Assigned here, deterministically:** ids, names, companies and email
  addresses, from fixed pools by position, so a rebuild names everything the
  same way.
- **Generated:** only the message text, by `writer.py`. Each message is
  checked against its spec (`checks.py`), and rewritten up to
  `MAX_ATTEMPTS` times. A spec that never passes stops the build; nothing is
  written.

The committed batch is the record. A rebuild calls the writer again and will
not reproduce the same wording, so an existing batch is never overwritten
without `--force`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.generate.checks import problems
from evals.generate.scenarios import BATCHES, TODAY, Spec
from evals.generate.writer import WRITER_MODEL, WRITER_TEMPERATURE, OpenAIWriter, Persona, Writer
from evals.loader import CUSTOMERS_PATH, POLICY_DIR, SYNTHETIC_CUSTOMERS, SYNTHETIC_DIR, load_synthetic
from evals.schema import Customer, Ticket, TicketLabels
from src.agent.openai_chat import DEFAULT_SEED
from src.agent.pricing import SpendGuard
from src.config import load_dotenv

MAX_ATTEMPTS = 3
DEFAULT_SPEND_CAP_USD = 1.00

#: (company, domain slug). Fictional; every domain is under `.example`.
COMPANIES = (
    ("Lark & Finch Studio", "larkfinch"), ("Northgate Analytics", "northgate"),
    ("Pebble Street Design", "pebblestreet"), ("Hollow Oak Games", "hollowoak"),
    ("Brightwater Legal Ops", "brightwater"), ("Cinder Labs", "cinderlabs"),
    ("Mosswood Architects", "mosswood"), ("Tidewell Logistics", "tidewell"),
    ("Quillon Publishing", "quillon"), ("Redfern Robotics", "redfern"),
    ("Saltbox Marketing", "saltbox"), ("Juniper & Vale", "junipervale"),
    ("Kestrel Health", "kestrelhealth"), ("Ironbridge Build", "ironbridge"),
    ("Orchard Row Media", "orchardrow"), ("Fennimore Events", "fennimore"),
    ("Glasswing Software", "glasswing"), ("Harrow & Pike", "harrowpike"),
    ("Lumen Field Research", "lumenfield"), ("Marlowe Consulting", "marlowe"),
    ("Nettlebed Farms", "nettlebed"), ("Oxbow Studios", "oxbow"),
    ("Pinecrest Schools", "pinecrest"), ("Rookery Records", "rookery"),
    ("Sable Coast Travel", "sablecoast"), ("Thistle Engineering", "thistle"),
    ("Umber Interiors", "umber"), ("Verity Audit", "verity"),
    ("Willowmere Clinic", "willowmere"), ("Yarrow Supply Co", "yarrow"),
)

PEOPLE = (
    ("Priya", "Shah"), ("Tom", "Okafor"), ("Lena", "Hartmann"), ("Diego", "Ramos"),
    ("Aisha", "Bello"), ("Marcus", "Lindqvist"), ("Yuki", "Tanaka"), ("Fatima", "Haddad"),
    ("Owen", "Gallagher"), ("Sofia", "Marchetti"), ("Kwame", "Mensah"), ("Hannah", "Weiss"),
    ("Ravi", "Iyer"), ("Chloe", "Dubois"), ("Mateo", "Fernandez"), ("Grace", "Nwosu"),
    ("Ivan", "Petrov"), ("Mei", "Lin"), ("Samuel", "Adeyemi"), ("Elena", "Kowalski"),
    ("Noah", "Brennan"), ("Zara", "Qureshi"), ("Lucas", "Moreau"), ("Ama", "Owusu"),
    ("Felix", "Bauer"), ("Nadia", "Rahman"), ("Jonah", "Silva"), ("Ines", "Costa"),
    ("Arjun", "Mehta"), ("Freya", "Olsen"), ("Kofi", "Asante"), ("Rosa", "Jimenez"),
    ("Theo", "Novak"), ("Leila", "Karimi"), ("Ben", "Whitaker"), ("Anya", "Volkova"),
    ("Hugo", "Laurent"), ("Maya", "Goldberg"), ("Emeka", "Obi"), ("Clara", "Lindgren"),
    ("Sami", "Nieminen"), ("Tara", "Doyle"), ("Kenji", "Mori"), ("Lucia", "Romano"),
    ("Dev", "Patel"), ("Ingrid", "Berg"), ("Omar", "Farouk"), ("Julia", "Kaminski"),
    ("Rhys", "Morgan"), ("Esi", "Boateng"),
)

ROLES = {
    "owner": "the account owner",
    "colleague": "an operations manager at the company, not the account owner",
    "member": "a team member on the account, not the account owner",
    "assistant": "the account owner's executive assistant",
    "former_employee": "a former employee who left the company in June",
    "impersonator": "someone writing as if they were the account owner, which they are not",
}


@dataclass(frozen=True)
class Assignment:
    ticket_id: str
    customer_id: str
    company: tuple[str, str]
    owner: tuple[str, str]
    owner_email: str
    requester_email: str
    persona: Persona


def assign(batch: int, specs: tuple[Spec, ...]) -> dict[str, Assignment]:
    """Ids, names and addresses for every spec, by position. A spec with
    `depends_on` shares its antecedent's customer and owner."""
    out: dict[str, Assignment] = {}
    for i, spec in enumerate(specs, 1):
        customer_id = f"CUST-{2000 + batch * 100 + i}"
        company = COMPANIES[(batch * 7 + i) % len(COMPANIES)]
        owner = PEOPLE[(batch * 11 + 2 * i) % len(PEOPLE)]
        other = PEOPLE[(batch * 11 + 2 * i + 1) % len(PEOPLE)]
        if spec.depends_on:
            earlier = out[spec.depends_on]
            customer_id, company, owner = earlier.customer_id, earlier.company, earlier.owner
        name, slug = company
        owner_email = _email(owner, f"{slug}.example")
        requester, address = _requester(spec.requester, owner, other, slug)
        out[spec.key] = Assignment(
            ticket_id=f"S-{batch:02d}{i:02d}",
            customer_id=customer_id,
            company=company,
            owner=owner,
            owner_email=owner_email,
            requester_email=address or owner_email,
            persona=Persona(f"{requester[0]} {requester[1]}", ROLES[spec.requester], name),
        )
    return out


def _in_batch(customer_id: str, batch: int) -> bool:
    """Synthetic customer ids are CUST-2000 + 100 * batch + position."""
    return int(customer_id.removeprefix("CUST-")) // 100 == 20 + batch


def _email(person: tuple[str, str], domain: str) -> str:
    return f"{person[0][0].lower()}.{person[1].lower()}@{domain}"


def _requester(kind: str, owner, other, slug: str) -> tuple[tuple[str, str], str | None]:
    """(name shown to the writer, address the ticket comes from). None means
    the owner's own address."""
    if kind == "owner":
        return owner, None
    if kind in ("colleague", "member", "assistant"):
        return other, _email(other, f"{slug}.example")
    if kind == "former_employee":
        return other, f"{other[0].lower()}{other[1].lower()}@mailbox.example"
    if kind == "impersonator":
        # The owner's name, on a lookalike domain.
        return owner, _email(owner, f"{slug}-team.example")
    raise ValueError(f"unknown requester kind {kind!r}")


def customer_record(spec: Spec, assignment: Assignment, batch: int, index: int) -> Customer:
    return Customer(
        customer_id=assignment.customer_id,
        owner_email=assignment.owner_email,
        plan=spec.plan,
        seats=spec.seats,
        billing_cycle=spec.billing_cycle,
        account_status=spec.account_status,
        last_charge_usd=spec.last_charge_usd,
        last_charge_date=spec.last_charge_date,
        past_tickets=[
            {"ticket_id": f"H-{batch}{index:02d}{j}", **past}
            for j, past in enumerate(spec.past_tickets, 1)
        ],
    )


def _same_account(a: Spec, b: Spec) -> bool:
    fields = ("plan", "seats", "billing_cycle", "account_status", "last_charge_usd", "last_charge_date")
    return all(getattr(a, f) == getattr(b, f) for f in fields) and not b.past_tickets


async def build_batch(
    batch: int,
    writer: Writer,
    *,
    synthetic_dir: Path = SYNTHETIC_DIR,
    customers_path: Path = CUSTOMERS_PATH,
    policy_dir: Path = POLICY_DIR,
    force: bool = False,
    seed: int = DEFAULT_SEED,
) -> Path:
    specs = BATCHES[batch]
    out = synthetic_dir / f"batch_{batch:02d}.json"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; it is the record of that batch. Pass --force to replace it.")
    keys = {s.key: s for s in specs}
    for spec in specs:
        if spec.depends_on and not _same_account(keys[spec.depends_on], spec):
            raise ValueError(f"{spec.key} depends on {spec.depends_on} but describes a different account")

    assignments = assign(batch, specs)
    entries, customers, served = [], {}, set()
    for index, spec in enumerate(specs, 1):
        a = assignments[spec.key]
        for attempt in range(MAX_ATTEMPTS):
            draft = await writer.write(spec, a.persona, seed + batch * 1000 + index * 10 + attempt)
            found = problems(spec, draft.body)
            if not found:
                break
        else:
            raise RuntimeError(f"{spec.key}: no message passed its checks in {MAX_ATTEMPTS} attempts: {found}")
        served.add(draft.served_model)
        ticket = Ticket(
            ticket_id=a.ticket_id,
            customer_id=a.customer_id,
            requester_email=a.requester_email,
            body=draft.body,
            labels=TicketLabels(
                outcome=spec.outcome,
                expected_tools=list(spec.expected_tools),
                relevant_policy_ids=list(spec.policy_ids),
                confidence=spec.confidence,
                label_note=f"Drafted: {spec.rationale}",
            ),
        )
        entries.append({
            "ticket": ticket.model_dump(mode="json"),
            "scenario": spec.key,
            "requester": spec.requester,
            "register": spec.register,
            "depends_on": assignments[spec.depends_on].ticket_id if spec.depends_on else None,
            "attempts": attempt + 1,
            "tags": list(spec.tags),
        })
        if not spec.depends_on:
            customers[a.customer_id] = customer_record(spec, a, batch, index).model_dump(mode="json")

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    customers_file = synthetic_dir / SYNTHETIC_CUSTOMERS
    existing = json.loads(customers_file.read_text(encoding="utf-8")) if customers_file.exists() else {}
    kept = {cid: rec for cid, rec in existing.items() if not _in_batch(cid, batch)}
    customers_file.write_text(json.dumps(kept | customers, indent=2) + "\n", encoding="utf-8")
    out.write_text(json.dumps({
        "batch": batch,
        "reviewed": False,
        "today": TODAY,
        "writer": {
            "model": WRITER_MODEL,
            "served_model": sorted(served),
            "temperature": WRITER_TEMPERATURE,
            "seed": seed,
        },
        "review": (
            "Check each ticket's labels against its body and the policy corpus: outcome, "
            "expected_tools, relevant_policy_ids. Edit anything wrong, replace the 'Drafted:' "
            "label_note with your own if you change a label, then set reviewed to true. "
            "Until then the batch is excluded from every eval run."
        ),
        "entries": entries,
    }, indent=2) + "\n", encoding="utf-8")

    # The batch must pass the same validation as a reviewed one would.
    load_synthetic(synthetic_dir, policy_dir, customers_path, include_unreviewed=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, required=True, choices=sorted(BATCHES))
    parser.add_argument("--force", action="store_true", help="replace an existing batch file")
    args = parser.parse_args(argv)
    load_dotenv()
    cap = float(os.environ.get("EVAL_SPEND_CAP_USD") or DEFAULT_SPEND_CAP_USD)
    spend = SpendGuard.for_model(WRITER_MODEL, cap)
    path = asyncio.run(build_batch(args.batch, OpenAIWriter(spend=spend), force=args.force))
    print(f"wrote {path} (writer spend ${spend.spent_usd:.4f}); review it, then set \"reviewed\": true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
