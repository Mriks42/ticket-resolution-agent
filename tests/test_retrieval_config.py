"""Choosing the retrieval backend, and fusing two of them.

The config rules here are the ones ADR 0003 makes mandatory: a dense backend
without a similarity floor is refused, because it would make "nothing in the
corpus covers this" inexpressible.
"""

from __future__ import annotations

import os

import pytest

from src.config import load_dotenv, read_dotenv
from src.tools.dense import DensePolicyIndex
from src.tools.policy import PolicyIndex, load_clauses
from src.tools.retrieval import (
    Backend,
    HybridPolicyIndex,
    RetrievalConfig,
    build_policy_index,
    rrf_fuse,
)

DB = "postgresql://tra:tra@localhost:5432/ticket_agent"


# --- RetrievalConfig.from_env ------------------------------------------------


def test_defaults_to_lexical() -> None:
    config = RetrievalConfig.from_env({})
    assert config.backend is Backend.LEXICAL
    assert config.dense_floor is None
    assert config.embedding_model == "text-embedding-3-small"


def test_empty_values_count_as_unset() -> None:
    """.env.example ships `RETRIEVAL_DENSE_FLOOR=` with nothing after it."""
    config = RetrievalConfig.from_env(
        {"RETRIEVAL_BACKEND": "", "RETRIEVAL_DENSE_FLOOR": "", "EMBEDDING_MODEL": ""}
    )
    assert config == RetrievalConfig()


def test_reads_every_setting() -> None:
    config = RetrievalConfig.from_env(
        {
            "RETRIEVAL_BACKEND": "dense",
            "RETRIEVAL_DENSE_FLOOR": "0.31",
            "EMBEDDING_MODEL": "text-embedding-3-large",
            "DATABASE_URL": DB,
        }
    )
    assert config.backend is Backend.DENSE
    assert config.dense_floor == 0.31
    assert config.embedding_model == "text-embedding-3-large"
    assert config.database_url == DB


def test_unknown_backend_is_rejected_naming_the_options() -> None:
    with pytest.raises(ValueError, match="lexical, dense, hybrid"):
        RetrievalConfig.from_env({"RETRIEVAL_BACKEND": "bm25"})


@pytest.mark.parametrize("backend", ["dense", "hybrid"])
def test_a_dense_backend_without_a_floor_is_refused(backend) -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_DENSE_FLOOR"):
        RetrievalConfig.from_env({"RETRIEVAL_BACKEND": backend, "DATABASE_URL": DB})


@pytest.mark.parametrize("backend", ["dense", "hybrid"])
def test_a_dense_backend_without_a_database_is_refused(backend) -> None:
    with pytest.raises(ValueError, match="DATABASE_URL"):
        RetrievalConfig.from_env(
            {"RETRIEVAL_BACKEND": backend, "RETRIEVAL_DENSE_FLOOR": "0.3"}
        )


@pytest.mark.parametrize("raw", ["high", "1.5", "-2", "nan"])
def test_a_floor_must_be_a_cosine_similarity(raw) -> None:
    with pytest.raises(ValueError, match="RETRIEVAL_DENSE_FLOOR"):
        RetrievalConfig.from_env({"RETRIEVAL_DENSE_FLOOR": raw})


def test_lexical_builds_the_lexical_index() -> None:
    index = build_policy_index(config=RetrievalConfig())
    assert isinstance(index, PolicyIndex)
    assert index.backend == "lexical"


# --- reciprocal rank fusion ----------------------------------------------------


def test_rrf_ranks_agreement_above_either_list_alone() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    assert [doc for doc, _ in fused][0] == "b"
    assert {doc for doc, _ in fused} == {"a", "b", "c", "d"}


def test_rrf_scores_are_the_sum_of_reciprocal_ranks() -> None:
    fused = dict(rrf_fuse([["a", "b"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_breaks_ties_on_id_so_runs_are_reproducible() -> None:
    fused = rrf_fuse([["z", "y"], ["y", "z"]], k=60)
    assert [doc for doc, _ in fused] == ["y", "z"]


def test_rrf_of_nothing_is_nothing() -> None:
    """Hybrid stays empty only when both backends are — which is what keeps
    an empty result reachable at all."""
    assert rrf_fuse([[], []]) == []


# --- hybrid over Postgres -------------------------------------------------------


@pytest.fixture
def hybrid(pg, hashing_embedder):
    clauses = load_clauses()
    dense = DensePolicyIndex(clauses, conn=pg, embedder=hashing_embedder, floor=0.5)
    return HybridPolicyIndex(PolicyIndex(clauses), dense)


def test_hybrid_returns_clause_level_hits_within_the_limit(hybrid) -> None:
    hits = hybrid.search("annual renewal charge refundable", limit=10)
    assert len(hits) <= 10
    assert "REF-001.3" in [c.clause_id for c, _ in hits]
    assert hybrid.backend == "hybrid"


def test_hybrid_is_empty_when_both_backends_are(hybrid) -> None:
    assert hybrid.search("zzzqqx nonexistent kumquat protocol", limit=10) == []


def test_hybrid_get_and_len(hybrid) -> None:
    assert len(hybrid) == 104
    assert hybrid.get("CAN-002.2").policy_id == "POL-CAN-002"


# --- .env --------------------------------------------------------------------


def test_read_dotenv_parses_comments_quotes_and_blanks(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nA=1\nB='two words'\nC=\"x\"\nD=\nnot a pair\n", encoding="utf-8"
    )
    assert read_dotenv(env) == {"A": "1", "B": "two words", "C": "x", "D": ""}


def test_read_dotenv_of_a_missing_file_is_empty(tmp_path) -> None:
    assert read_dotenv(tmp_path / "absent") == {}


def test_load_dotenv_never_overrides_the_environment(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("A=from-file\nB=from-file\n", encoding="utf-8")
    environ = {"A": "from-env"}
    monkeypatch.setattr(os, "environ", environ)

    load_dotenv(env)

    assert environ == {"A": "from-env", "B": "from-file"}
