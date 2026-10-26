"""The synthetic generator, without a network call.

The writer model is the one thing faked: `FakeWriter` returns a message that
carries the spec's required facts. Everything else runs for real — the specs,
the checks, id and address assignment, the batch file, and the loader that
decides what may enter an eval. The spec tests pin the label conventions in
`scenarios.py`, so a spec that drifts from them fails before any money is
spent writing it.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from datetime import date

import pytest

from evals.generate.build import build_batch
from evals.generate.checks import problems
from evals.generate.scenarios import BATCH_1, BATCHES, REGISTERS, REQUESTERS, Spec
from evals.generate.writer import Draft
from evals.loader import CUSTOMERS_PATH, POLICY_DIR, load_policy_ids, load_synthetic
from evals.generate.scenarios import TODAY
from evals.schema import Outcome, Tool
from src.agent.gate import recent_refunds as recent
from src.tools.context import ToolContext
from src.tools.policy import load_clauses
from src.tools.retrieval import RetrievalConfig

#: Every spec in every batch: the conventions hold for all of them.
ALL_SPECS = [spec for batch in BATCHES.values() for spec in batch]

FILLER = "Hello, could you help with this for us, thank you very much for taking a look at it today."


class FakeWriter:
    def __init__(self, body=None) -> None:
        self.body = body
        self.seeds: list[int] = []

    async def write(self, spec: Spec, persona, seed: int) -> Draft:
        self.seeds.append(seed)
        if self.body is not None:
            return Draft(body=self.body, served_model="fake")
        facts = " ".join(options[0] for options in spec.must_mention)
        return Draft(body=f"{FILLER} {facts}", served_model="fake")


def build(tmp_path, writer=None, **kw):
    return asyncio.run(build_batch(1, writer or FakeWriter(), synthetic_dir=tmp_path, **kw))


def entries(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


# --- the specs follow the label conventions ---------------------------------------


def test_spec_keys_registers_and_requesters_are_valid() -> None:
    assert len({s.key for s in ALL_SPECS}) == len(ALL_SPECS)
    assert all(len(batch) == 25 for batch in BATCHES.values())
    for spec in ALL_SPECS:
        assert spec.register in REGISTERS, spec.key
        assert spec.requester in REQUESTERS, spec.key


def test_every_cited_policy_and_clause_exists() -> None:
    policies, clauses = load_policy_ids(), {c.clause_id for c in load_clauses()}
    for spec in ALL_SPECS:
        assert set(spec.policy_ids) <= policies, spec.key
        cited = set(re.findall(r"\b[A-Z]{3}-\d{3}\.\d+\b", spec.rationale))
        assert cited <= clauses, (spec.key, cited - clauses)


def recent_refunds(spec: Spec) -> int:
    return recent(spec.refunds, date.fromisoformat(TODAY))


def test_a_held_refund_is_an_owner_in_window_refund_the_gate_holds() -> None:
    """Above the self-serve ceiling, or under it with two refunds in the
    preceding 12 months (REF-002.4)."""
    for spec in ALL_SPECS:
        if spec.outcome is Outcome.PENDING_APPROVAL:
            assert spec.requester == "owner", spec.key
            assert spec.last_charge_usd > 50 or recent_refunds(spec) >= 2, spec.key
            assert spec.expected_tools[-1] is Tool.ISSUE_REFUND, spec.key
            assert any("day" in o or "week" in o for opts in spec.must_mention for o in opts), (
                f"{spec.key}: the agent has no clock, so the message must say how long ago"
            )


def test_a_refund_that_resolves_is_one_the_gate_executes() -> None:
    for spec in ALL_SPECS:
        if spec.outcome is Outcome.RESOLVE and Tool.ISSUE_REFUND in spec.expected_tools:
            assert spec.last_charge_usd <= 50, spec.key
            assert recent_refunds(spec) < 2, spec.key


def test_refund_history_sits_well_inside_or_outside_the_window() -> None:
    """The gate uses the real date and the eval world's is 2026-08-24 (ADR
    0011): a refund must count the same under both, months either way."""
    for spec in ALL_SPECS:
        for later in ("2026-08-24", "2026-12-31"):
            assert recent(spec.refunds, date.fromisoformat(later)) == recent_refunds(spec), spec.key


def test_a_non_owner_resolves_only_a_general_question() -> None:
    for spec in ALL_SPECS:
        if spec.requester != "owner" and spec.outcome is not Outcome.ESCALATE:
            assert set(spec.expected_tools) == {Tool.RETRIEVE_POLICY, Tool.SEND_REPLY}, spec.key


def test_history_dependent_specs_carry_their_history() -> None:
    for batch in BATCHES.values():
        keys = [s.key for s in batch]
        for spec in batch:
            if "history" in spec.tags and "antecedent" not in spec.tags:
                assert Tool.SEARCH_PAST_TICKETS in spec.expected_tools, spec.key
                assert spec.past_tickets or spec.depends_on, spec.key
            if spec.depends_on:
                assert keys.index(spec.depends_on) < keys.index(spec.key), spec.key


def test_the_messy_end_is_sampled_deliberately() -> None:
    for batch in BATCHES.values():
        rough = sum(s.register in ("messy", "buried") for s in batch)
        assert rough / len(batch) >= 0.4


# --- checks -----------------------------------------------------------------------


SPEC = next(s for s in BATCH_1 if s.key == "H-04")  # must have $88, must not mention the bank


def test_a_message_carrying_its_facts_passes() -> None:
    assert problems(SPEC, f"{FILLER} Can you refund the $88 from the 2nd?") == []


@pytest.mark.parametrize(("body", "expected"), [
    (f"{FILLER} Can you refund the charge from the 2nd?", "missing one of"),
    (f"{FILLER} Refund the $88, my bank dispute is still open.", "contains 'bank'"),
    (f"{FILLER} Refund the $88 under REF-001.1 please.", "policy or clause id"),
    ("Refund the $88.", "words"),
])
def test_a_message_missing_or_leaking_a_fact_fails(body, expected) -> None:
    assert any(expected in p for p in problems(SPEC, body))


# --- the build --------------------------------------------------------------------------


def test_the_batch_is_written_unreviewed_with_its_customers(tmp_path) -> None:
    path = build(tmp_path)
    batch = json.loads(path.read_text(encoding="utf-8"))
    assert batch["reviewed"] is False
    assert [e["ticket"]["ticket_id"] for e in batch["entries"]] == [f"S-01{i:02d}" for i in range(1, 26)]
    customers = json.loads((tmp_path / "customers.json").read_text(encoding="utf-8"))
    assert len(customers) == 24, "H-04 shares H-03's customer"
    assert all(e["ticket"]["labels"]["label_note"].startswith("Drafted: ") for e in batch["entries"])


def test_requester_addresses_follow_who_is_writing(tmp_path) -> None:
    path = build(tmp_path)
    customers = json.loads((tmp_path / "customers.json").read_text(encoding="utf-8"))
    for e in entries(path):
        owner = customers[e["ticket"]["customer_id"]]["owner_email"]
        sent = e["ticket"]["requester_email"]
        if e["requester"] == "owner":
            assert sent == owner
        else:
            assert sent != owner
        if e["requester"] == "impersonator":
            assert sent.split("@")[0] == owner.split("@")[0], "the owner's name"
            assert sent.split("@")[1] != owner.split("@")[1], "on another domain"


def test_a_dependent_ticket_shares_its_antecedents_customer_and_runs_after_it(tmp_path) -> None:
    by_key = {e["scenario"]: e for e in entries(build(tmp_path))}
    earlier, later = by_key["H-03"], by_key["H-04"]
    assert later["depends_on"] == earlier["ticket"]["ticket_id"]
    assert later["ticket"]["customer_id"] == earlier["ticket"]["customer_id"]


def test_fixture_history_lands_in_the_customer_record(tmp_path) -> None:
    by_key = {e["scenario"]: e for e in entries(build(tmp_path))}
    customers = json.loads((tmp_path / "customers.json").read_text(encoding="utf-8"))
    past = customers[by_key["H-01"]["ticket"]["customer_id"]]["past_tickets"]
    assert "chargeback" in past[0]["summary"]


def test_billing_history_lands_in_the_customer_record(tmp_path) -> None:
    by_key = {e["scenario"]: e for e in entries(build(tmp_path))}
    customers = json.loads((tmp_path / "customers.json").read_text(encoding="utf-8"))
    assert customers[by_key["H-02"]["ticket"]["customer_id"]]["refunds"] == [
        {"date": "2026-08-19", "amount_usd": 176.0, "ticket_id": None},
    ]


def test_an_unreviewed_batch_never_enters_an_eval(tmp_path) -> None:
    path = build(tmp_path)
    assert load_synthetic(tmp_path) == []
    assert len(load_synthetic(tmp_path, include_unreviewed=True)) == 25
    batch = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**batch, "reviewed": True}), encoding="utf-8")
    assert len(load_synthetic(tmp_path)) == 25


def test_an_existing_batch_is_not_overwritten_without_force(tmp_path) -> None:
    build(tmp_path)
    with pytest.raises(FileExistsError, match="--force"):
        build(tmp_path)
    build(tmp_path, force=True)
    assert len(json.loads((tmp_path / "customers.json").read_text(encoding="utf-8"))) == 24


def test_a_writer_that_never_passes_the_checks_stops_the_build(tmp_path) -> None:
    writer = FakeWriter(body="too short")
    with pytest.raises(RuntimeError, match="P-01"):
        build(tmp_path, writer=writer)
    assert len(writer.seeds) == 3
    assert not (tmp_path / "batch_01.json").exists()


def test_the_tools_serve_synthetic_customers(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(POLICY_DIR, dataset / "policies")
    shutil.copy(CUSTOMERS_PATH, dataset / "customers.json")
    path = build(dataset / "synthetic")
    first = entries(path)[0]["ticket"]
    ctx = ToolContext.build(var_dir=tmp_path / "var", dataset_dir=dataset, retrieval=RetrievalConfig())
    assert ctx.fixtures.customer(first["customer_id"])["customer_id"] == first["customer_id"]
    assert ctx.fixtures.customer("CUST-1042")["plan"] == "pro", "held-out customers still there"


# --- the loader's rules for a synthetic batch ---------------------------------------------


def doctor(tmp_path, change) -> None:
    path = build(tmp_path)
    batch = json.loads(path.read_text(encoding="utf-8"))
    change({e["scenario"]: e for e in batch["entries"]})
    path.write_text(json.dumps(batch), encoding="utf-8")


def test_a_non_owner_ticket_cannot_resolve_with_a_refund(tmp_path) -> None:
    doctor(tmp_path, lambda by: by["N-04"]["ticket"]["labels"]["expected_tools"].append("issue_refund"))
    with pytest.raises(ValueError, match="cannot expect a refund"):
        load_synthetic(tmp_path, include_unreviewed=True)


def test_a_non_owner_ticket_that_resolves_must_say_why(tmp_path) -> None:
    doctor(tmp_path, lambda by: by["N-04"]["ticket"]["labels"].update(label_note=None))
    with pytest.raises(ValueError, match="need a label_note"):
        load_synthetic(tmp_path, include_unreviewed=True)


def test_a_dependency_must_run_first_for_the_same_customer(tmp_path) -> None:
    doctor(tmp_path, lambda by: by["H-04"].update(depends_on="S-0125"))
    with pytest.raises(ValueError, match="does not run before it"):
        load_synthetic(tmp_path, include_unreviewed=True)
