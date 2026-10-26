"""The six tools, exercised directly against the fixture corpus.

These deliberately pin values that can be checked by hand against
`evals/dataset/`. A tool that starts inventing plausible-looking data at call
time instead of reading the fixtures fails here.
"""

from __future__ import annotations

import json

import pytest

from evals.loader import load_tickets
from src.memory.close import Close
from src.tools.catalog import CATALOG, Scope
from src.tools.context import ToolContext
from src.tools.errors import ToolInputError
from src.tools.policy import load_clauses
from src.tools.retrieval import RetrievalConfig
from src.tools.store import WriteStore


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    """A context whose writes land in an isolated var dir.

    Pinned to the lexical backend: the retrieval assertions below are about
    lexical behaviour, and must not change with a developer's environment.
    """
    return ToolContext.build(var_dir=tmp_path / "var", retrieval=RetrievalConfig())


def call(ctx: ToolContext, tool: str, **args):
    return CATALOG[tool].handler(ctx, **args)


# --- catalog ---------------------------------------------------------------


def test_catalog_holds_exactly_the_six_tools() -> None:
    assert set(CATALOG) == {
        "lookup_customer",
        "search_past_tickets",
        "retrieve_policy",
        "issue_refund",
        "send_reply",
        "escalate",
    }


def test_write_tools_are_marked_gated_and_reads_are_not() -> None:
    gated = {name for name, spec in CATALOG.items() if spec.gated}
    assert gated == {"issue_refund", "send_reply", "escalate"}


def test_every_tool_declares_a_scope() -> None:
    """No tool may fall back to an open default."""
    for name, spec in CATALOG.items():
        assert isinstance(spec.scope, Scope), name


def test_read_and_write_tools_do_not_share_a_scope() -> None:
    """Granting reads must not transitively grant a write."""
    read_scopes = {s.scope for s in CATALOG.values() if not s.gated}
    write_scopes = {s.scope for s in CATALOG.values() if s.gated}
    assert not (read_scopes & write_scopes)


# --- lookup_customer -------------------------------------------------------


def test_lookup_customer_returns_the_fixture_record(ctx: ToolContext) -> None:
    result = call(ctx, "lookup_customer", customer_id="CUST-1042")
    assert result["customer_id"] == "CUST-1042"
    assert result["owner_email"] == "r.mehta@brightfold.example"
    assert result["plan"] == "pro"
    assert result["last_charge_usd"] == 29.0


def test_lookup_customer_rejects_an_unknown_id(ctx: ToolContext) -> None:
    with pytest.raises(ToolInputError, match="CUST-9999"):
        call(ctx, "lookup_customer", customer_id="CUST-9999")


# --- search_past_tickets ---------------------------------------------------


def test_search_past_tickets_scopes_to_one_customer(ctx: ToolContext) -> None:
    result = call(
        ctx,
        "search_past_tickets",
        query="slow response missed target",
        customer_id="CUST-1520",
    )
    ids = [m["ticket_id"] for m in result["matches"]]
    assert set(ids) <= {"H-4471", "H-5088"}
    assert ids, "CUST-1520 has two prior contacts about response times"


def test_search_past_tickets_returns_empty_for_a_customer_with_no_history(
    ctx: ToolContext,
) -> None:
    """Empty is a real answer, not an error. CUST-1042 has no past_tickets."""
    result = call(ctx, "search_past_tickets", query="anything", customer_id="CUST-1042")
    assert result["matches"] == []


def test_search_past_tickets_without_customer_id_searches_all(
    ctx: ToolContext,
) -> None:
    result = call(ctx, "search_past_tickets", query="unexplained line item invoice")
    ids = [m["ticket_id"] for m in result["matches"]]
    assert "H-4623" in ids


def test_search_past_tickets_rejects_an_unknown_customer(ctx: ToolContext) -> None:
    with pytest.raises(ToolInputError, match="CUST-9999"):
        call(ctx, "search_past_tickets", query="anything", customer_id="CUST-9999")


def test_search_past_tickets_returns_history_even_when_no_word_matches(ctx: ToolContext) -> None:
    """No floor (ADR 0008): empty means "never written in", so a customer with
    history gets it back however the query is phrased."""
    result = call(ctx, "search_past_tickets", query="zzzqqx kumquat", customer_id="CUST-1520")
    assert [m["ticket_id"] for m in result["matches"]] == ["H-4471", "H-5088"]
    assert list(result["matches"][0]) == [
        "ticket_id", "customer_id", "opened", "summary", "resolution", "score",
    ]


def test_lookup_customer_carries_the_history_memory_holds(ctx: ToolContext) -> None:
    record = call(ctx, "lookup_customer", customer_id="CUST-1520")
    assert [p["ticket_id"] for p in record["past_tickets"]] == ["H-4471", "H-5088"]
    assert list(record["past_tickets"][0]) == ["ticket_id", "opened", "summary", "resolution"]
    assert list(record)[-1] == "past_tickets", "same shape the model saw before memory existed"


def test_lookup_customer_shows_remembered_facts_once_there_are_some(ctx: ToolContext) -> None:
    """Absent until the customer has a close, so the account record is
    exactly what it was before memory existed for everyone else."""
    assert "memory" not in call(ctx, "lookup_customer", customer_id="CUST-1042")
    written = ctx.memory.remember(Close(
        run_id="run-1", customer_id="CUST-1042", ticket_id="T-001", body="Charged twice.",
        final_state="ESCALATE", date="2026-10-26", clauses=("BIL-001.2",),
    ))
    record = call(ctx, "lookup_customer", customer_id="CUST-1042")
    assert record["memory"] == written["facts"]
    assert list(record)[-2:] == ["past_tickets", "memory"]


def test_what_one_context_remembers_another_does_not_see(tmp_path) -> None:
    """One context per eval run, so one run's writes never reach the next."""
    a = ToolContext.build(var_dir=tmp_path / "a", retrieval=RetrievalConfig())
    b = ToolContext.build(var_dir=tmp_path / "b", retrieval=RetrievalConfig())
    a.memory.remember(Close(
        run_id="run-1", customer_id="CUST-1042", ticket_id="T-001", body="Charged twice.",
        final_state="ESCALATE", date="2026-10-26",
    ))
    assert [p["ticket_id"] for p in call(a, "lookup_customer", customer_id="CUST-1042")["past_tickets"]] == ["T-001"]
    assert call(b, "lookup_customer", customer_id="CUST-1042")["past_tickets"] == []
    assert a.memory.namespace != b.memory.namespace


# --- retrieve_policy -------------------------------------------------------


def test_corpus_parses_into_clauses_with_clause_level_ids() -> None:
    clauses = load_clauses()
    assert len(clauses) == 104
    by_id = {c.clause_id: c for c in clauses}
    assert by_id["REF-001.3"].policy_id == "POL-REF-001"
    assert "14 days" in by_id["REF-001.3"].text


def test_clause_ids_are_unique_across_the_corpus() -> None:
    clauses = load_clauses()
    assert len({c.clause_id for c in clauses}) == len(clauses)


def test_retrieve_policy_returns_clause_ids_not_document_ids(
    ctx: ToolContext,
) -> None:
    """The Session 6 guardrail needs the specific clause, not the document."""
    result = call(ctx, "retrieve_policy", query="annual renewal charge refundable")
    assert result["passages"], "expected at least one passage"
    for passage in result["passages"]:
        # REF-001.3, never POL-REF-001.
        assert passage["clause_id"].count(".") == 1
        assert not passage["clause_id"].startswith("POL-")
        assert passage["policy_id"].startswith("POL-")


def test_retrieve_policy_finds_the_annual_renewal_clause(ctx: ToolContext) -> None:
    result = call(ctx, "retrieve_policy", query="annual renewal charge refundable")
    assert "REF-001.3" in [p["clause_id"] for p in result["passages"]]


def test_retrieve_policy_surfaces_both_sides_of_the_planted_conflict(
    ctx: ToolContext,
) -> None:
    """POLICY_NOTES records REF-001.3 vs CAN-002.2 as a deliberate conflict.

    T-016 tests whether the agent notices it, which it cannot do if retrieval
    can only ever reach one side. This pins that both clauses are reachable
    from one query -- an index property.

    It does *not* show that a customer's own phrasing reaches both. It does
    not: "annual auto renewed 10 days ago want money back" ranks CAN-002.2
    fourth and misses REF-001.3 entirely, because the lexical backend needs
    shared vocabulary and the customer has none of it. That gap is a property
    of the retriever, not of the corpus, and it should show up as a
    hallucinated-policy or escalation number once the agent loop exists rather
    than being tuned away here.
    """
    result = call(
        ctx,
        "retrieve_policy",
        query="annual subscription refund after renewal term begun",
        limit=12,
    )
    found = {p["clause_id"] for p in result["passages"]}
    assert {"REF-001.3", "CAN-002.2"} <= found


def test_default_limit_reaches_both_sides_of_the_conflict_for_t016(
    ctx: ToolContext,
) -> None:
    """The default `limit` must not hide the contradiction T-016 tests for.

    Against T-016's real body, REF-001.3 ranks 1st and CAN-002.2 ranks 6th, so
    the old default of 5 returned one side and looked confident about it. This
    pins the default high enough to surface both, using the ticket text rather
    than a paraphrase -- testing a paraphrase is what produced the wrong
    diagnosis in the first place. See ADR 0003.
    """
    t016 = next(t for t in load_tickets() if t.ticket_id == "T-016")
    result = call(ctx, "retrieve_policy", query=t016.body)
    found = {p["clause_id"] for p in result["passages"]}
    assert {"REF-001.3", "CAN-002.2"} <= found


def test_retrieve_policy_returns_nothing_when_nothing_matches(
    ctx: ToolContext,
) -> None:
    """A deliberate corpus gap must read as empty, not as a weak best guess."""
    result = call(ctx, "retrieve_policy", query="zzzqqx nonexistent kumquat protocol")
    assert result["passages"] == []


def test_retrieve_policy_respects_the_limit(ctx: ToolContext) -> None:
    result = call(ctx, "retrieve_policy", query="refund", limit=3)
    assert len(result["passages"]) <= 3


def test_retrieve_policy_is_deterministic(ctx: ToolContext) -> None:
    a = call(ctx, "retrieve_policy", query="seat removal mid cycle billing")
    b = call(ctx, "retrieve_policy", query="seat removal mid cycle billing")
    assert a == b


# --- write tools -----------------------------------------------------------


def test_issue_refund_writes_to_the_store(ctx: ToolContext) -> None:
    result = call(
        ctx,
        "issue_refund",
        ticket_id="T-001",
        amount=29.0,
        reason="within the 30-day window per REF-001.1",
    )
    assert result["gated"] is True
    records = ctx.store.records("refunds")
    assert len(records) == 1
    assert records[0]["ticket_id"] == "T-001"
    assert records[0]["amount"] == 29.0


def test_issue_refund_rejects_a_non_positive_amount(ctx: ToolContext) -> None:
    with pytest.raises(ToolInputError, match="amount"):
        call(ctx, "issue_refund", ticket_id="T-001", amount=0, reason="nope")
    assert ctx.store.records("refunds") == []


def test_send_reply_writes_to_the_store(ctx: ToolContext) -> None:
    result = call(ctx, "send_reply", ticket_id="T-002", body="Refund is on its way.")
    assert result["gated"] is True
    records = ctx.store.records("replies")
    assert [r["body"] for r in records] == ["Refund is on its way."]


def test_send_reply_rejects_an_empty_body(ctx: ToolContext) -> None:
    with pytest.raises(ToolInputError, match="body"):
        call(ctx, "send_reply", ticket_id="T-002", body="   ")


def test_escalate_writes_the_trace_it_was_given(ctx: ToolContext) -> None:
    trace = [{"state": "PLAN", "note": "two clauses conflict"}]
    call(
        ctx,
        "escalate",
        ticket_id="T-016",
        reason="REF-001.3 and CAN-002.2 disagree",
        trace=trace,
    )
    records = ctx.store.records("escalations")
    assert records[0]["trace"] == trace
    assert records[0]["reason"] == "REF-001.3 and CAN-002.2 disagree"


def test_store_is_append_only(ctx: ToolContext) -> None:
    call(ctx, "send_reply", ticket_id="T-001", body="first")
    first_pass = (ctx.store.path("replies")).read_text(encoding="utf-8")
    call(ctx, "send_reply", ticket_id="T-002", body="second")
    second_pass = (ctx.store.path("replies")).read_text(encoding="utf-8")
    assert second_pass.startswith(first_pass)
    assert len(ctx.store.records("replies")) == 2


def test_store_writes_one_json_object_per_line(ctx: ToolContext) -> None:
    call(ctx, "send_reply", ticket_id="T-001", body="first")
    call(ctx, "send_reply", ticket_id="T-002", body="second")
    lines = ctx.store.path("replies").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)


def test_var_dir_is_isolated_per_context(tmp_path) -> None:
    """Eval runs and tests must not write into each other's store."""
    a = ToolContext.build(var_dir=tmp_path / "a")
    b = ToolContext.build(var_dir=tmp_path / "b")
    a.store.append("replies", {"ticket_id": "T-001", "body": "only in a"})
    assert len(a.store.records("replies")) == 1
    assert b.store.records("replies") == []


def test_write_store_stays_out_of_the_dataset(ctx: ToolContext, tmp_path) -> None:
    call(ctx, "issue_refund", ticket_id="T-001", amount=29.0, reason="x")
    assert (tmp_path / "var") in ctx.store.path("refunds").parents


def test_read_tools_do_not_write_anything(ctx: ToolContext) -> None:
    call(ctx, "lookup_customer", customer_id="CUST-1042")
    call(ctx, "retrieve_policy", query="refund window")
    call(ctx, "search_past_tickets", query="invoice")
    for kind in ("refunds", "replies", "escalations"):
        assert ctx.store.records(kind) == []


def test_unknown_store_kind_is_rejected(ctx: ToolContext) -> None:
    """Guards against a typo silently creating a new store file."""
    with pytest.raises(ValueError, match="unknown store"):
        ctx.store.append("refundz", {"oops": True})
