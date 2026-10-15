"""The HITL gate's rules, as pure decisions.

No loop, no model, no tools: given an action and the facts the gate holds,
what does it decide? These are the deterministic rules CLAUDE.md asks for —
action type and retrieval coverage, never model confidence.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.agent.gate import (
    MANAGER_REFUND_MAX,
    SELF_SERVE_REFUND_MAX,
    Verdict,
    cited_clause_ids,
    decide_refund,
    decide_reply,
)
from src.tools.policy import load_clauses

OWNER = "r.mehta@brightfold.example"


def refund(amount, *, already=0, requester=OWNER, owner=OWNER, cited=(), retrieved=()):
    return decide_refund(
        amount=amount,
        already_refunded=already,
        requester_email=requester,
        owner_email=owner,
        cited=set(cited),
        retrieved=set(retrieved),
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


# --- coverage: every cited clause must have been retrieved ---------------------------


def test_a_refund_reason_citing_an_unretrieved_clause_escalates() -> None:
    decision = refund(29, cited={"REF-001.1"}, retrieved={"REF-001.2"})
    assert decision.verdict is Verdict.ESCALATE
    assert decision.rule == "citation_not_retrieved"
    assert "REF-001.1" in decision.detail


def test_a_reply_citing_retrieved_clauses_executes() -> None:
    decision = decide_reply(cited={"REF-001.1"}, retrieved={"REF-001.1", "REF-001.2"}, pending=False)
    assert decision.verdict is Verdict.EXECUTE


def test_a_reply_citing_an_unretrieved_clause_is_a_hallucinated_policy_event() -> None:
    decision = decide_reply(cited={"REF-001.1", "CAN-002.2"}, retrieved={"REF-001.1"}, pending=False)
    assert decision.verdict is Verdict.ESCALATE
    assert decision.rule == "citation_not_retrieved"
    assert "CAN-002.2" in decision.detail


def test_a_reply_with_no_citations_executes() -> None:
    """Uncited policy claims are the guardrail's to catch, not the gate's."""
    assert decide_reply(cited=set(), retrieved=set(), pending=False).verdict is Verdict.EXECUTE


def test_a_reply_while_an_approval_is_pending_is_held_with_it() -> None:
    decision = decide_reply(cited=set(), retrieved=set(), pending=True)
    assert decision.verdict is Verdict.HOLD
    assert decision.rule == "approval_pending"


def test_coverage_is_checked_before_a_pending_hold() -> None:
    """A hallucinated citation is never parked for approval."""
    decision = decide_reply(cited={"REF-001.9"}, retrieved=set(), pending=True)
    assert decision.verdict is Verdict.ESCALATE


# --- citations --------------------------------------------------------------------


def test_only_full_clause_ids_count_as_citations() -> None:
    text = "Per REF-001.3 and (CAN-002.2). See POL-REF-001 and ESC-001 generally."
    assert cited_clause_ids(text) == {"REF-001.3", "CAN-002.2"}
