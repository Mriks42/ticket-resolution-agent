"""The HITL gate's rules, as pure decisions.

No loop, no model, no tools: given an action and the facts the gate holds,
what does it decide? These are the action-type rules CLAUDE.md asks for,
never model confidence. Retrieval coverage is the guardrail's, in
`test_guardrail.py`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.agent.gate import (
    MANAGER_REFUND_MAX,
    REFUND_HISTORY_DAYS,
    REFUND_HISTORY_LIMIT,
    SELF_SERVE_REFUND_MAX,
    Verdict,
    decide_refund,
    decide_reply,
    recent_refunds,
)
from src.tools.policy import load_clauses

OWNER = "r.mehta@brightfold.example"


def refund(amount, *, already=0, requester=OWNER, owner=OWNER, prior=0):
    return decide_refund(
        amount=amount, already_refunded=already, requester_email=requester, owner_email=owner,
        prior_refunds=prior,
    )


# --- refund bands: ESC-001.1-.3 as arithmetic --------------------------------


@pytest.mark.parametrize(
    "amount, verdict, clause, approver",
    [
        (29, Verdict.EXECUTE, "ESC-001.1", None),
        (50, Verdict.EXECUTE, "ESC-001.1", None),
        (50.01, Verdict.HOLD, "ESC-001.2", "support manager"),
        (176, Verdict.HOLD, "ESC-001.2", "support manager"),
        (500, Verdict.HOLD, "ESC-001.2", "support manager"),
        (500.01, Verdict.HOLD, "ESC-001.3", "finance"),
        (1584, Verdict.HOLD, "ESC-001.3", "finance"),
    ],
)
def test_refund_bands(amount, verdict, clause, approver) -> None:
    decision = refund(amount)
    assert decision.verdict is verdict
    assert decision.clause == clause
    assert decision.approver == approver


def test_band_thresholds_are_the_ones_the_policy_states() -> None:
    """The constants are policy, copied into code. If ESC-001 changes and the
    code does not, this fails rather than the gate silently disagreeing."""
    text = {c.clause_id: c.text for c in load_clauses()}
    assert f"up to USD {int(SELF_SERVE_REFUND_MAX)} without approval" in text["ESC-001.1"]
    assert (
        f"between USD {SELF_SERVE_REFUND_MAX + Decimal('0.01')} "
        f"and USD {int(MANAGER_REFUND_MAX)}"
    ) in text["ESC-001.2"]
    assert f"above USD {int(MANAGER_REFUND_MAX)}" in text["ESC-001.3"]


def test_bands_apply_to_the_total_refunded_on_the_ticket() -> None:
    """Decided with the label owner: ESC-001.1's limit is per ticket, so a
    refund cannot be split into pieces that each look self-serve."""
    assert refund(30).verdict is Verdict.EXECUTE
    second = refund(25, already=30)
    assert second.verdict is Verdict.HOLD
    assert second.clause == "ESC-001.2"
    assert "55.00" in second.detail


def test_split_refunds_cannot_add_up_past_the_self_serve_limit() -> None:
    assert refund(50, already=0).verdict is Verdict.EXECUTE
    assert refund(0.01, already=50).verdict is Verdict.HOLD


def test_the_total_can_cross_into_the_finance_band() -> None:
    decision = refund(300, already=300)
    assert decision.clause == "ESC-001.3"
    assert decision.approver == "finance"


def test_amounts_are_compared_in_cents() -> None:
    assert refund(50.004).verdict is Verdict.EXECUTE
    assert refund(50.006).verdict is Verdict.HOLD


# --- the owner check: ESC-001.4, money only --------------------------------------


def test_a_non_owner_refund_escalates_whatever_the_amount() -> None:
    decision = refund(10, requester="colleague@brightfold.example")
    assert decision.verdict is Verdict.ESCALATE
    assert decision.clause == "ESC-001.4"
    assert decision.rule == "non_owner"


def test_the_owner_comparison_ignores_case_and_whitespace() -> None:
    assert refund(10, requester="  R.Mehta@Brightfold.example ").verdict is Verdict.EXECUTE


def test_an_owner_that_cannot_be_verified_fails_closed() -> None:
    decision = refund(10, owner=None)
    assert decision.verdict is Verdict.ESCALATE
    assert decision.rule == "owner_unverified"


def test_the_owner_check_comes_before_the_band() -> None:
    """A non-owner asking for $1,584 escalates; it does not wait for finance."""
    assert refund(1584, requester="x@y.example").verdict is Verdict.ESCALATE


# --- refund history: REF-002.4 ---------------------------------------------------------


def test_two_prior_refunds_hold_even_a_self_serve_amount() -> None:
    decision = refund(20, prior=2)
    assert (decision.verdict, decision.rule, decision.clause, decision.approver) == (
        Verdict.HOLD, "refund_history", "REF-002.4", "support manager",
    )


def test_one_prior_refund_changes_nothing() -> None:
    assert refund(20, prior=1).clause == "ESC-001.1"


def test_refund_history_never_lowers_a_band() -> None:
    assert refund(176, prior=2).clause == "ESC-001.2"
    assert refund(600, prior=3).approver == "finance"


def test_the_owner_check_comes_before_refund_history() -> None:
    assert refund(20, prior=5, requester="x@y.example").rule == "non_owner"


def test_the_refund_history_limit_is_the_one_the_policy_states() -> None:
    text = {c.clause_id: c.text for c in load_clauses()}["REF-002.4"]
    assert "two or more refunds in the preceding 12 months" in text
    assert (REFUND_HISTORY_LIMIT, REFUND_HISTORY_DAYS) == (2, 365)


TODAY = date(2026, 8, 24)


def test_recent_refunds_counts_the_preceding_twelve_months() -> None:
    history = [
        {"date": "2026-08-24", "amount_usd": 10.0},  # today
        {"date": "2025-08-25", "amount_usd": 10.0},  # 364 days ago
        {"date": "2025-08-24", "amount_usd": 10.0},  # 365 days ago: outside
        {"date": "2024-01-01", "amount_usd": 10.0},
    ]
    assert recent_refunds(history, TODAY) == 2


def test_a_refund_with_no_readable_date_counts() -> None:
    """Failing closed: an unreadable record holds a refund, never frees one."""
    assert recent_refunds([{"amount_usd": 10.0}, {"date": "last spring"}], TODAY) == 2


# --- replies ------------------------------------------------------------------------


def test_a_reply_executes_when_nothing_is_pending() -> None:
    assert decide_reply(pending=False).verdict is Verdict.EXECUTE


def test_a_reply_while_an_approval_is_pending_is_held_with_it() -> None:
    decision = decide_reply(pending=True)
    assert decision.verdict is Verdict.HOLD
    assert decision.rule == "approval_pending"
