"""The guardrail: deterministic checks of a reply's policy facts.

Pure functions, no loop. Every fact a reply states — an amount, a duration, a
count, a percentage — must be grounded in a passage retrieved in the run or in
the customer's own data; a fact in a sentence that cites a clause must be in
*that* clause. Qualitative claims carry no checkable fact and are not caught;
one test below pins that blind spot so it stays visible.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.agent.guardrail import Fact, Sources, check, cited_clause_ids, extract_facts, split_sentences
from src.tools.policy import load_clauses

TEXT = {c.clause_id: c.text for c in load_clauses()}


def passages(*clause_ids: str) -> dict[str, str]:
    return {cid: TEXT[cid] for cid in clause_ids}


def kinds(reply: str, sources: Sources) -> list[str]:
    return [v.kind for v in check(reply, sources)]


def facts(text: str) -> set[tuple[str, Decimal, str]]:
    return {(f.kind, f.value, f.unit) for f in extract_facts(text)}


# --- extraction --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("up to USD 50 without approval", {("money", Decimal("50.00"), "usd")}),
        ("a $1,584 renewal", {("money", Decimal("1584.00"), "usd")}),
        ("between USD 50.01 and USD 500", {("money", Decimal("50.01"), "usd"), ("money", Decimal("500.00"), "usd")}),
        ("a fee of 15 dollars", {("money", Decimal("15.00"), "usd")}),
        ("within 14 days of the renewal date", {("duration", Decimal("14"), "day")}),
        ("a 30-day window", {("duration", Decimal("30"), "day")}),
        ("two weeks", {("duration", Decimal("14"), "day")}),
        ("settle in 5–10 business days", {("duration", Decimal("5"), "business_day"), ("duration", Decimal("10"), "business_day")}),
        ("within 4 business hours", {("duration", Decimal("4"), "business_hour")}),
        ("more than 60 consecutive days", {("duration", Decimal("60"), "day")}),
        ("the preceding 12 months", {("duration", Decimal("12"), "month")}),
        ("a minimum of 3 billable seats", {("count", Decimal("3"), "seat")}),
        ("at the 3-seat minimum", {("count", Decimal("3"), "seat")}),
        ("up to 3 projects, 500 MB of storage", {("count", Decimal("3"), "project"), ("storage", Decimal("500"), "mb")}),
        ("50 GB of storage", {("storage", Decimal("50"), "gb")}),
        ("a 20% discount", {("percent", Decimal("20"), "percent")}),
    ],
)
def test_facts_are_extracted_and_normalised(text, expected) -> None:
    assert facts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "see REF-001.3 and POL-ESC-001",
        "invoice INV-88213 on the 2nd",
        "charged on 13 August 2026",
        "if the day-7 retry fails",
        "Settings → Export, in JSON and CSV formats",
    ],
)
def test_ids_dates_and_ordinals_are_not_facts(text) -> None:
    assert facts(text) == set()


def test_sentences_do_not_split_on_decimals_or_clause_ids() -> None:
    reply = "Refunds of USD 50.01 or more need approval (ESC-001.2). We will be in touch."
    assert split_sentences(reply) == [
        "Refunds of USD 50.01 or more need approval (ESC-001.2).",
        "We will be in touch.",
    ]


# --- grounding ----------------------------------------------------------------------


def test_a_fact_from_the_cited_clause_is_clean() -> None:
    sources = Sources(passages=passages("REF-001.1"))
    assert kinds("Monthly plans are refundable within 30 days (REF-001.1).", sources) == []


def test_an_uncited_fact_from_any_retrieved_passage_is_clean() -> None:
    sources = Sources(passages=passages("REF-001.5"))
    assert kinds("Refunds settle in 5–10 business days.", sources) == []


def test_equivalent_units_match() -> None:
    sources = Sources(passages=passages("REF-001.3"))
    assert kinds("Renewals can be refunded within two weeks (REF-001.3).", sources) == []


def test_the_customers_own_figures_are_grounded() -> None:
    sources = Sources(
        passages=passages("REF-001.1"),
        context=("Our monthly charge of $176 for the eight seats went through on 13 August",),
    )
    assert kinds("Your $176 charge is within the 30-day window (REF-001.1).", sources) == []


def test_an_amount_the_agent_proposed_is_grounded() -> None:
    sources = Sources(passages={}, context_facts=frozenset({Fact("money", Decimal("132.00"), "usd", "$132")}))
    assert kinds("A refund of $132 has been submitted for approval.", sources) == []


# --- violations -------------------------------------------------------------------------


def test_a_wrong_figure_in_a_cited_sentence_is_unsupported() -> None:
    sources = Sources(passages=passages("REF-001.1"))
    [v] = check("Monthly plans are refundable within 45 days (REF-001.1).", sources)
    assert v.kind == "unsupported_fact"
    assert "45 days" in v.detail


def test_a_figure_from_the_wrong_clause_is_miscited() -> None:
    """30 days is real policy — but REF-001.1's, not REF-001.3's."""
    sources = Sources(passages=passages("REF-001.1", "REF-001.3"))
    [v] = check("Annual renewals are refundable within 30 days (REF-001.3).", sources)
    assert v.kind == "miscited_fact"
    assert "REF-001.3" in v.detail


def test_an_invented_fee_is_unsupported() -> None:
    sources = Sources(passages=passages("REF-001.5"))
    assert kinds("Refunds carry a $15 processing fee.", sources) == ["unsupported_fact"]


def test_an_invented_discount_is_unsupported() -> None:
    assert kinds("We can offer you 20% off next month.", Sources(passages={})) == ["unsupported_fact"]


def test_a_wrong_count_is_unsupported() -> None:
    sources = Sources(passages=passages("SEA-002.1", "SEA-001.3"))
    assert kinds("Team needs a minimum of 5 seats (SEA-002.1).", sources) == ["unsupported_fact"]


def test_citing_an_unretrieved_clause_is_a_violation() -> None:
    sources = Sources(passages=passages("REF-001.1"))
    [v] = check("Renewals are refundable (REF-001.3).", sources)
    assert v.kind == "citation_not_retrieved"
    assert "REF-001.3" in v.detail


def test_each_bad_fact_is_reported() -> None:
    sources = Sources(passages=passages("REF-001.1"))
    reply = "You have 45 days to ask (REF-001.1). There is also a $15 fee."
    assert kinds(reply, sources) == ["unsupported_fact", "unsupported_fact"]


# --- the documented blind spot -------------------------------------------------------------


def test_a_qualitative_claim_is_not_caught() -> None:
    """Known limitation, decided with the label owner: no number, nothing to
    check. The reviewed reply set measures how much this misses."""
    sources = Sources(passages=passages("CAN-002.2"))
    assert kinds("Annual plans can be refunded at any time.", sources) == []


def test_only_full_clause_ids_count_as_citations() -> None:
    text = "Per REF-001.3 and (CAN-002.2). See POL-REF-001 and ESC-001 generally."
    assert cited_clause_ids(text) == {"REF-001.3", "CAN-002.2"}
