"""The pgvector-backed policy index.

These run against real Postgres in a throwaway schema (see `conftest.pg`) with
a deterministic hashing embedder, so they exercise the SQL, the sync, the
cache and the similarity floor without an API call. What they cannot tell you
is whether the real model ranks well — that is `evals/retrieval_baseline.py`'s
job, and it only ever uses the real model.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.tools.dense import DensePolicyIndex
from src.tools.errors import ToolError
from src.tools.policy import indexed_text, load_clauses
from tests.conftest import HashingEmbedder

OFF_TOPIC = "zzzqqx nonexistent kumquat protocol"


@pytest.fixture(scope="module")
def clauses():
    return load_clauses()


def build(pg, clauses, embedder, floor=None) -> DensePolicyIndex:
    return DensePolicyIndex(clauses, conn=pg, embedder=embedder, floor=floor)


# --- the retrieve_policy contract ------------------------------------------


def test_hits_are_clause_level(pg, clauses, hashing_embedder) -> None:
    index = build(pg, clauses, hashing_embedder)
    hits = index.search("annual renewal charge refundable", limit=10)
    assert "REF-001.3" in [c.clause_id for c, _ in hits]
    for clause, _ in hits:
        assert not clause.clause_id.startswith("POL-")
        assert clause.policy_id.startswith("POL-")


def test_limit_is_respected(pg, clauses, hashing_embedder) -> None:
    index = build(pg, clauses, hashing_embedder)
    assert len(index.search("refund", limit=3)) == 3


def test_scores_are_cosine_similarity_best_first(pg, clauses, hashing_embedder) -> None:
    """A query identical to a clause's indexed text is that clause, at ~1.0."""
    index = build(pg, clauses, hashing_embedder)
    target = next(c for c in clauses if c.clause_id == "SEA-002.2")
    hits = index.search(indexed_text(target), limit=5)

    assert hits[0][0].clause_id == "SEA-002.2"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_ranking_is_deterministic(pg, clauses, hashing_embedder) -> None:
    index = build(pg, clauses, hashing_embedder)
    query = "seat removal mid cycle billing"
    assert index.search(query, limit=20) == index.search(query, limit=20)


def test_empty_query_returns_nothing_and_costs_nothing(pg, clauses, hashing_embedder) -> None:
    index = build(pg, clauses, hashing_embedder)
    before = hashing_embedder.texts_embedded
    assert index.search("   ", limit=5) == []
    assert hashing_embedder.texts_embedded == before


def test_get_and_len(pg, clauses, hashing_embedder) -> None:
    index = build(pg, clauses, hashing_embedder)
    assert len(index) == len(clauses)
    assert index.get("REF-001.3").policy_id == "POL-REF-001"
    assert index.get("NOPE-000.1") is None


# --- the similarity floor --------------------------------------------------


def test_without_a_floor_dense_always_returns_a_nearest_neighbour(
    pg, clauses, hashing_embedder
) -> None:
    """The regression ADR 0003 warns about, pinned so it stays visible: with
    no floor, nonsense still retrieves `limit` passages."""
    index = build(pg, clauses, hashing_embedder, floor=None)
    assert len(index.search(OFF_TOPIC, limit=5)) == 5


def test_floor_turns_an_off_topic_query_into_an_empty_result(
    pg, clauses, hashing_embedder
) -> None:
    index = build(pg, clauses, hashing_embedder, floor=0.5)
    assert index.search(OFF_TOPIC, limit=5) == []


def test_floor_keeps_hits_at_or_above_it(pg, clauses, hashing_embedder) -> None:
    unfloored = build(pg, clauses, hashing_embedder).search("annual renewal refund", limit=20)
    cut = unfloored[3][1]
    floored = build(pg, clauses, hashing_embedder, floor=cut).search(
        "annual renewal refund", limit=20
    )
    assert floored == [(c, s) for c, s in unfloored if s >= cut]
    assert len(floored) >= 4


# --- sync and caching ------------------------------------------------------


def test_corpus_is_embedded_once_across_builds(pg, clauses, hashing_embedder) -> None:
    build(pg, clauses, hashing_embedder)
    assert hashing_embedder.texts_embedded == len(clauses)

    build(pg, clauses, hashing_embedder)
    assert hashing_embedder.texts_embedded == len(clauses)


def test_a_changed_clause_is_re_embedded_and_nothing_else(
    pg, clauses, hashing_embedder
) -> None:
    build(pg, clauses, hashing_embedder)
    edited = [
        replace(c, text="Seat changes are billed on the kumquat anniversary.")
        if c.clause_id == "SEA-002.2"
        else c
        for c in clauses
    ]
    index = build(pg, edited, hashing_embedder)

    assert hashing_embedder.texts_embedded == len(clauses) + 1
    assert index.search("kumquat anniversary", limit=1)[0][0].clause_id == "SEA-002.2"


def test_a_removed_clause_is_never_returned(pg, clauses, hashing_embedder) -> None:
    build(pg, clauses, hashing_embedder)
    remaining = [c for c in clauses if c.clause_id != "REF-001.3"]
    index = build(pg, remaining, hashing_embedder)

    everything = index.search("annual renewal charge refundable", limit=len(clauses))
    assert "REF-001.3" not in {c.clause_id for c, _ in everything}
    assert len(everything) == len(remaining)


def test_a_repeated_query_is_served_from_the_cache(pg, clauses, hashing_embedder) -> None:
    """Eval reruns must make no API calls and rank identically, which is what
    keeps a published number reproducible after the fact."""
    index = build(pg, clauses, hashing_embedder)
    first = index.search("downgrade from team to free", limit=10)
    after_first = hashing_embedder.texts_embedded

    rebuilt = build(pg, clauses, hashing_embedder)
    second = rebuilt.search("downgrade from team to free", limit=10)

    assert hashing_embedder.texts_embedded == after_first
    assert second == first
    assert rebuilt.stats.query_cache_hits == 1


def test_models_never_share_vectors(pg, clauses) -> None:
    """Rankings under different models are not comparable (ADR 0003), so a
    vector cached under one model must never answer for another."""
    a = HashingEmbedder(model="model-a")
    b = HashingEmbedder(model="model-b", dims=128)
    build(pg, clauses, a).search("refund window", limit=3)
    build(pg, clauses, b).search("refund window", limit=3)

    assert a.texts_embedded == len(clauses) + 1
    assert b.texts_embedded == len(clauses) + 1


# --- failure handling ------------------------------------------------------


def test_a_database_failure_mid_run_is_a_tool_error(pg, clauses, hashing_embedder) -> None:
    """A lost connection narrows the agent's options; it must reach OBSERVE
    as a tool error, not crash the run."""
    index = build(pg, clauses, hashing_embedder)
    pg.close()
    with pytest.raises(ToolError, match="policy search failed"):
        index.search("refund window", limit=3)
