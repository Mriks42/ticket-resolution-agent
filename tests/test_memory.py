"""The memory layer's two stores, against both backends.

One contract per layer, two implementations each (ADR 0008): in-process for
the lexical configuration, Postgres for dense and hybrid. Each contract test
runs against both; the Postgres ones use a throwaway schema and are skipped
without `docker compose up -d`. The episodic embedder is the hashing one, so
those tests pin the plumbing — scoping, ordering, namespaces, the cache — not
how well a real model ranks. `fold`, which turns a close into durable facts,
is pure and tested on its own.
"""

from __future__ import annotations

import pytest

from dataclasses import replace

from src.memory import Memory
from src.memory.close import Close, Refund
from src.memory.episodic import Episode, InProcessEpisodes, PgEpisodes, resolution_text
from src.memory.semantic import InProcessFacts, PgFacts, fold

SEED = [
    Episode("CUST-A", "H-1", "2026-06-01", "Charged twice for June.", "Refunded the duplicate."),
    Episode("CUST-A", "H-2", "2026-07-01", "Asked how to export a project.", "Sent the export guide."),
    Episode("CUST-B", "H-3", "2026-06-15", "Charged twice on the annual plan.", "Escalated to billing."),
]


@pytest.fixture(params=["in-process", "pgvector"])
def make_store(request):
    if request.param == "in-process":
        return lambda namespace="ns-1": InProcessEpisodes(SEED, namespace=namespace)
    pg = request.getfixturevalue("pg")
    embedder = request.getfixturevalue("hashing_embedder")
    return lambda namespace="ns-1": PgEpisodes(SEED, conn=pg, embedder=embedder, namespace=namespace)


def ids(hits) -> list[str]:
    return [e.ticket_id for e, _ in hits]


# --- the contract ------------------------------------------------------------


def test_history_is_the_customers_own_oldest_first(make_store) -> None:
    store = make_store()
    assert [e.ticket_id for e in store.history("CUST-A")] == ["H-1", "H-2"]
    assert store.history("CUST-NONE") == []


def test_search_ranks_the_customers_history_best_first(make_store) -> None:
    hits = make_store().search("charged twice", "CUST-A", 5)
    assert ids(hits) == ["H-1", "H-2"]
    assert hits[0][1] > hits[1][1]


def test_search_has_no_floor_so_empty_means_no_history(make_store) -> None:
    """An off-topic query still returns the customer's history: an empty list
    tells the agent the customer has never written in, so it must mean that."""
    store = make_store()
    assert set(ids(store.search("zzzqqx kumquat", "CUST-A", 5))) == {"H-1", "H-2"}
    assert set(ids(store.search("", "CUST-A", 5))) == {"H-1", "H-2"}
    assert store.search("charged twice", "CUST-NONE", 5) == []


def test_search_without_a_customer_searches_everyone(make_store) -> None:
    assert "H-3" in ids(make_store().search("charged twice annual plan", None, 5))


def test_search_respects_the_limit(make_store) -> None:
    assert len(make_store().search("charged", "CUST-A", 1)) == 1


def test_an_added_episode_is_in_history_and_search(make_store) -> None:
    store = make_store()
    store.add(Episode("CUST-A", "T-9", "2026-10-26", "Charged twice again.", "Escalated to a human."))
    assert [e.ticket_id for e in store.history("CUST-A")] == ["H-1", "H-2", "T-9"]
    assert "T-9" in ids(store.search("charged twice", "CUST-A", 5))


def test_adding_the_same_ticket_again_replaces_it(make_store) -> None:
    store = make_store()
    store.add(Episode("CUST-A", "T-9", "2026-10-26", "First.", "Escalated to a human."))
    store.add(Episode("CUST-A", "T-9", "2026-10-26", "Second.", "Resolved by the agent."))
    assert [e.summary for e in store.history("CUST-A") if e.ticket_id == "T-9"] == ["Second."]


def test_namespaces_do_not_see_each_others_writes(make_store) -> None:
    """Each eval run gets its own namespace, so one run's writes never reach
    the next and a rerun starts from the same state."""
    a, b = make_store("ns-a"), make_store("ns-b")
    a.add(Episode("CUST-A", "T-9", "2026-10-26", "Only in a.", "Escalated to a human."))
    assert "T-9" not in [e.ticket_id for e in b.history("CUST-A")]
    assert "T-9" not in ids(b.search("only in a", "CUST-A", 5))
    assert [e.ticket_id for e in b.history("CUST-A")] == ["H-1", "H-2"]


# --- pgvector only -------------------------------------------------------------


def test_seed_history_is_embedded_once_across_namespaces(pg, hashing_embedder) -> None:
    PgEpisodes(SEED, conn=pg, embedder=hashing_embedder, namespace="ns-a")
    PgEpisodes(SEED, conn=pg, embedder=hashing_embedder, namespace="ns-b")
    assert hashing_embedder.texts_embedded == len(SEED)


# --- what an agent-written episode holds ------------------------------------------


def close(final_state: str, **kw) -> Close:
    return Close(
        run_id="run-1", customer_id="CUST-A", ticket_id="T-9", body="I was charged twice.",
        final_state=final_state, date="2026-10-26", **kw,
    )


def test_an_episode_is_the_customers_words_and_a_fixed_outcome() -> None:
    episode = Episode.from_close(close("RESOLVE", clauses=("REF-001.3",)))
    assert episode == Episode(
        "CUST-A", "T-9", "2026-10-26", "I was charged twice.",
        "Resolved by the agent; a reply was sent. Clauses cited: REF-001.3.",
    )


@pytest.mark.parametrize(("state", "opening"), [
    ("RESOLVE", "Resolved by the agent; a reply was sent."),
    ("PENDING_APPROVAL", "Waiting on an approver."),
    ("ESCALATE", "Escalated to a human."),
    ("FAILED", "Escalated to a human; the agent could not complete the run."),
])
def test_every_terminal_state_has_an_outcome(state, opening) -> None:
    assert resolution_text(close(state)) == opening


def test_refunds_are_recorded_with_their_status() -> None:
    text = resolution_text(close(
        "PENDING_APPROVAL", refunds=(Refund(29.0, "issued"), Refund(290.0, "held")),
    ))
    assert text == (
        "Waiting on an approver. Refund of $29.00 issued. Refund of $290.00 held for approval."
    )


def test_an_unknown_terminal_state_is_refused() -> None:
    with pytest.raises(ValueError, match="PLAN"):
        resolution_text(close("PLAN"))


# --- semantic memory: folding a close into the customer's facts --------------------


def test_the_first_close_starts_the_record() -> None:
    facts = fold({}, close("ESCALATE", clauses=("BIL-001.2", "ESC-001.6")))
    assert facts == {
        "tickets_closed": 1,
        "refunds": [],
        "escalations": [{"ticket_id": "T-9", "date": "2026-10-26", "clauses": ["BIL-001.2", "ESC-001.6"]}],
        "policies_cited": {"POL-BIL-001": 1, "POL-ESC-001": 1},
        "recurring": [],
    }


def test_refunds_keep_their_status_and_a_pending_ticket_is_not_an_escalation() -> None:
    facts = fold({}, close(
        "PENDING_APPROVAL", refunds=(Refund(29.0, "issued"), Refund(290.0, "held")),
    ))
    assert facts["refunds"] == [
        {"ticket_id": "T-9", "amount": 29.0, "status": "issued", "date": "2026-10-26"},
        {"ticket_id": "T-9", "amount": 290.0, "status": "held", "date": "2026-10-26"},
    ]
    assert facts["escalations"] == []


def test_a_failed_run_counts_as_an_escalation() -> None:
    assert len(fold({}, close("FAILED"))["escalations"]) == 1


def test_a_policy_cited_at_two_closes_is_recurring_but_escalation_procedure_is_not() -> None:
    """Counted once per ticket however many of its clauses are cited."""
    first = fold({}, close("ESCALATE", clauses=("SLA-001.3", "SLA-001.5", "ESC-001.6")))
    second = fold(first, close("ESCALATE", clauses=("SLA-001.5", "ESC-001.6")))
    assert first["policies_cited"]["POL-SLA-001"] == 1
    assert second["policies_cited"] == {"POL-ESC-001": 2, "POL-SLA-001": 2}
    assert second["recurring"] == ["POL-SLA-001"]
    assert second["tickets_closed"] == 2


def test_fold_does_not_change_its_input() -> None:
    first = fold({}, close("ESCALATE", refunds=(Refund(29.0, "issued"),)))
    snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in first.items()}
    fold(first, close("ESCALATE", refunds=(Refund(10.0, "issued"),)))
    assert first == snapshot


# --- semantic memory: the store contract -------------------------------------------


@pytest.fixture(params=["in-process", "postgres"])
def make_facts(request):
    if request.param == "in-process":
        return lambda namespace="ns-1": InProcessFacts(namespace=namespace)
    pg = request.getfixturevalue("pg")
    return lambda namespace="ns-1": PgFacts(conn=pg, namespace=namespace)


def test_a_customer_with_no_closes_has_no_facts(make_facts) -> None:
    assert make_facts().get("CUST-A") == {}


def test_facts_come_back_as_written_key_order_included(make_facts) -> None:
    """Key order reaches the model in the account record, so it must not
    depend on the backend."""
    store = make_facts()
    facts = fold({}, close("PENDING_APPROVAL", refunds=(Refund(290.0, "held"),), clauses=("ESC-001.2",)))
    store.put("CUST-A", facts)
    back = store.get("CUST-A")
    assert back == facts
    assert list(back) == list(facts)
    assert list(back["refunds"][0]) == list(facts["refunds"][0])


def test_facts_namespaces_do_not_see_each_others_writes(make_facts) -> None:
    a, b = make_facts("ns-a"), make_facts("ns-b")
    a.put("CUST-A", fold({}, close("ESCALATE")))
    assert b.get("CUST-A") == {}


def test_remember_writes_both_layers() -> None:
    memory = Memory(InProcessEpisodes(SEED, namespace="ns-1"), InProcessFacts(namespace="ns-1"))
    first = memory.remember(close("ESCALATE", clauses=("SLA-001.5",)))
    second = memory.remember(replace(close("ESCALATE", clauses=("SLA-001.5",)), ticket_id="T-10"))
    assert first["facts"]["tickets_closed"] == 1
    assert second["facts"]["recurring"] == ["POL-SLA-001"]
    assert memory.facts.get("CUST-A") == second["facts"]
    assert [e.ticket_id for e in memory.episodes.history("CUST-A")][-2:] == ["T-9", "T-10"]
