"""Checks that the harness reports what actually happened.

The point of these is adversarial: a harness that always prints plausible
numbers is worse than no harness. Each test pins a value that can be derived
by hand from the dataset, so a silent change in metric definitions fails here.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from evals.agents.stub import AlwaysEscalateAgent
from evals.loader import load_customers, load_policy_ids, load_tickets
from evals.metrics import bootstrap_ci, evaluate
from evals.run import run_agent
from evals.schema import AgentResult, Confidence, Outcome, Ticket, Tool, ToolCall


class GoldAgent:
    """Replays the labels. Upper bound on every metric."""

    name = "gold"

    async def run(self, ticket: Ticket) -> AgentResult:
        return AgentResult(
            ticket_id=ticket.ticket_id,
            run_id=f"gold-{uuid4().hex[:8]}",
            outcome=ticket.labels.outcome,
            tool_calls=[ToolCall(tool=t) for t in ticket.labels.expected_tools],
        )


class ResolveEverythingAgent:
    name = "resolve-everything"

    async def run(self, ticket: Ticket) -> AgentResult:
        return AgentResult(
            ticket_id=ticket.ticket_id,
            run_id=f"ra-{uuid4().hex[:8]}",
            outcome=Outcome.RESOLVE,
            tool_calls=[],
        )


@pytest.fixture(scope="module")
def tickets() -> list[Ticket]:
    return load_tickets()


def score(agent, tickets: list[Ticket]) -> dict[str, object]:
    results = asyncio.run(run_agent(agent, tickets))
    return {m.name: m for m in evaluate(tickets, results, resamples=2000)}


def test_corpus_and_dataset_load(tickets: list[Ticket]) -> None:
    assert len(load_policy_ids()) == 20
    assert len(tickets) == 22
    assert len({t.ticket_id for t in tickets}) == 22


def test_gold_distribution_is_what_the_labels_say(tickets: list[Ticket]) -> None:
    counts = {o: sum(1 for t in tickets if t.labels.outcome is o) for o in Outcome}
    assert counts[Outcome.RESOLVE] == 10
    assert counts[Outcome.ESCALATE] == 9
    assert counts[Outcome.PENDING_APPROVAL] == 3


def test_uncertain_labels_are_flagged_for_review(tickets: list[Ticket]) -> None:
    drafts = [t for t in tickets if t.labels.confidence is not Confidence.HIGH]
    assert [t.ticket_id for t in drafts] == ["T-017"]
    assert all(t.labels.label_note for t in drafts)


def test_search_past_tickets_is_only_expected_where_history_exists(
    tickets: list[Ticket],
) -> None:
    customers = load_customers()
    expecting = [
        t for t in tickets if Tool.SEARCH_PAST_TICKETS in t.labels.expected_tools
    ]
    assert {t.ticket_id for t in expecting} == {"T-009", "T-015", "T-020"}
    for ticket in expecting:
        assert customers[ticket.customer_id].past_tickets


def test_loader_rejects_search_past_tickets_without_history(tmp_path) -> None:
    """The guard must fail a bad label, not merely pass the current data."""
    doctored = tmp_path / "tickets.json"
    doctored.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "T-900",
                    # CUST-1042 has no past_tickets in the fixtures.
                    "customer_id": "CUST-1042",
                    "body": "Where are my invoices?",
                    "labels": {
                        "outcome": "resolve",
                        "expected_tools": ["search_past_tickets", "send_reply"],
                        "relevant_policy_ids": ["POL-BIL-001"],
                        "confidence": "high",
                        "label_note": None,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no prior history"):
        load_tickets(doctored)


def test_stub_scores_the_hand_derived_floor(tickets: list[Ticket]) -> None:
    m = score(AlwaysEscalateAgent(), tickets)
    # Escalates all 22: never correct on the 10 resolve + 3 pending_approval.
    assert m["resolution_rate"].value == 0.0
    assert m["pending_approval_rate"].value == 0.0
    # 9 of 22 escalations match a gold escalate label.
    assert m["escalation_precision"].value == pytest.approx(9 / 22)
    # One escalate() call per ticket; escalate is an expected tool on exactly
    # the 9 gold-escalate tickets.
    assert m["tool_call_precision"].value == pytest.approx(9 / 22)


def test_zero_scores_carry_a_degenerate_ci(tickets: list[Ticket]) -> None:
    m = score(AlwaysEscalateAgent(), tickets)
    assert m["resolution_rate"].is_degenerate
    assert m["escalation_precision"].is_degenerate is False


def test_gold_agent_hits_the_ceiling(tickets: list[Ticket]) -> None:
    m = score(GoldAgent(), tickets)
    assert m["resolution_rate"].value == pytest.approx(10 / 22)
    assert m["escalation_precision"].value == pytest.approx(1.0)
    assert m["tool_call_precision"].value == pytest.approx(1.0)
    assert m["pending_approval_rate"].value == pytest.approx(3 / 22)


def test_escalation_precision_is_undefined_not_zero(tickets: list[Ticket]) -> None:
    m = score(ResolveEverythingAgent(), tickets)
    assert m["escalation_precision"].denominator == 0
    assert m["escalation_precision"].value is None
    assert m["tool_call_precision"].value is None
    # Resolving everything is right on exactly the 10 gold-resolve tickets.
    assert m["resolution_rate"].value == pytest.approx(10 / 22)


def test_agent_returning_wrong_ticket_id_fails_loudly(tickets: list[Ticket]) -> None:
    class LiarAgent:
        name = "liar"

        async def run(self, ticket: Ticket) -> AgentResult:
            return AgentResult(
                ticket_id="T-999",
                run_id="x",
                outcome=Outcome.RESOLVE,
            )

    with pytest.raises(ValueError, match="T-999"):
        asyncio.run(run_agent(LiarAgent(), tickets))


def test_bootstrap_ci_brackets_the_point_estimate(tickets: list[Ticket]) -> None:
    for metric in score(AlwaysEscalateAgent(), tickets).values():
        if metric.value is None:
            continue
        assert metric.ci_low <= metric.value <= metric.ci_high


def test_bootstrap_is_seed_deterministic() -> None:
    pairs = [(1.0, 1.0)] * 7 + [(0.0, 1.0)] * 13
    assert bootstrap_ci(pairs, resamples=500, seed=1) == bootstrap_ci(
        pairs, resamples=500, seed=1
    )
    assert bootstrap_ci(pairs, resamples=500, seed=1) != bootstrap_ci(
        pairs, resamples=500, seed=2
    )
