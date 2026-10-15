"""Which policy index `retrieve_policy` searches, chosen by config.

Three backends, which together are the retrieval ablation axis ADR 0003 adds:

- **lexical** — TF-IDF (`src/tools/lexical.py`). No dependencies; a clause
  sharing no word with the query is not returned.
- **dense** — cosine similarity over pgvector (`src/tools/dense.py`), with a
  mandatory similarity floor.
- **hybrid** — reciprocal rank fusion of the two.

Lexical is the default. It stays the default until
`python -m evals.retrieval_baseline` shows another backend beating it, and the
README does not get to claim otherwise before that number exists.

Every backend satisfies `PolicySearch`, so the catalog, the MCP server and the
agent loop are indifferent to which one is behind `retrieve_policy`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from src.tools.dense import DensePolicyIndex
from src.tools.embeddings import DEFAULT_EMBEDDING_MODEL, OpenAIEmbedder
from src.tools.policy import (
    DEFAULT_POLICY_DIR,
    Clause,
    PolicyIndex,
    load_clauses,
    load_policy_index,
)

#: The constant from Cormack et al. (2009), used as-is. RRF is insensitive to
#: it over a wide range, and tuning it on 22 tickets would be fitting noise.
DEFAULT_RRF_K = 60


class Backend(str, Enum):
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


class PolicySearch(Protocol):
    backend: str

    def search(self, query: str, limit: int = ...) -> list[tuple[Clause, float]]: ...

    def get(self, clause_id: str) -> Clause | None: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class RetrievalConfig:
    backend: Backend = Backend.LEXICAL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dense_floor: float | None = None
    database_url: str | None = None

    def __post_init__(self) -> None:
        # Written so NaN fails too: every comparison against NaN is False.
        if self.dense_floor is not None and not -1.0 <= self.dense_floor <= 1.0:
            raise ValueError(
                f"RETRIEVAL_DENSE_FLOOR is a cosine similarity and must be in "
                f"[-1, 1]; got {self.dense_floor!r}"
            )
        if self.backend is Backend.LEXICAL:
            return
        if self.dense_floor is None:
            raise ValueError(
                f"RETRIEVAL_DENSE_FLOOR is required for the {self.backend.value} "
                "backend. Without a floor dense retrieval always returns a nearest "
                "neighbour, and 'no policy covers this' becomes inexpressible "
                "(ADR 0003)."
            )
        if not self.database_url:
            raise ValueError(
                f"DATABASE_URL is required for the {self.backend.value} backend; "
                "run `docker compose up -d` and set it in .env."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RetrievalConfig:
        env = os.environ if env is None else env

        def get(name: str) -> str:
            return (env.get(name) or "").strip()

        raw_backend = get("RETRIEVAL_BACKEND") or Backend.LEXICAL.value
        try:
            backend = Backend(raw_backend)
        except ValueError:
            options = ", ".join(b.value for b in Backend)
            raise ValueError(
                f"RETRIEVAL_BACKEND must be one of {options}; got {raw_backend!r}"
            ) from None

        raw_floor = get("RETRIEVAL_DENSE_FLOOR")
        try:
            floor = float(raw_floor) if raw_floor else None
        except ValueError:
            raise ValueError(
                f"RETRIEVAL_DENSE_FLOOR must be a number in [-1, 1]; got {raw_floor!r}"
            ) from None

        return cls(
            backend=backend,
            embedding_model=get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            dense_floor=floor,
            database_url=get("DATABASE_URL") or None,
        )

    def describe(self) -> dict:
        """For eval reports. Never includes the database URL, which may carry
        a password."""
        record = asdict(self)
        record["backend"] = self.backend.value
        record.pop("database_url")
        if self.backend is Backend.LEXICAL:
            record.pop("embedding_model")
            record.pop("dense_floor")
        return record


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = DEFAULT_RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each id scores the sum of 1 / (k + rank) over
    the rankings it appears in. Best first, ties broken by id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridPolicyIndex:
    """Lexical and dense rankings, fused with RRF.

    Each backend contributes everything it would return at all — every clause
    sharing a word with the query, every clause above the dense floor — and the
    fused list is cut to `limit`. So hybrid is empty only when both backends
    are, and the score is an RRF score, not a similarity.
    """

    backend = "hybrid"

    def __init__(
        self, lexical: PolicyIndex, dense: DensePolicyIndex, *, k: int = DEFAULT_RRF_K
    ) -> None:
        if len(lexical) != len(dense):
            raise ValueError(
                f"hybrid needs both backends over the same corpus: lexical has "
                f"{len(lexical)} clauses, dense has {len(dense)}"
            )
        self._lexical = lexical
        self._dense = dense
        self._k = k

    @property
    def dense(self) -> DensePolicyIndex:
        return self._dense

    def search(self, query: str, limit: int = 5) -> list[tuple[Clause, float]]:
        depth = len(self._dense)
        fused = rrf_fuse(
            [
                [c.clause_id for c, _ in self._lexical.search(query, depth)],
                [c.clause_id for c, _ in self._dense.search(query, depth)],
            ],
            k=self._k,
        )
        return [(self._dense.get(clause_id), score) for clause_id, score in fused[:limit]]

    def get(self, clause_id: str) -> Clause | None:
        return self._dense.get(clause_id)

    def __len__(self) -> int:
        return len(self._dense)


def build_policy_index(
    policy_dir: Path = DEFAULT_POLICY_DIR, config: RetrievalConfig | None = None
) -> PolicySearch:
    """The index `retrieve_policy` should search under `config` (default: the
    environment). Dense and hybrid connect to Postgres and embed any clause not
    yet stored, so building one is where a missing database or API key fails —
    loudly, before the first ticket rather than in the middle of one."""
    config = config or RetrievalConfig.from_env()
    if config.backend is Backend.LEXICAL:
        return load_policy_index(policy_dir)

    import psycopg

    dense = DensePolicyIndex(
        load_clauses(policy_dir),
        conn=psycopg.connect(config.database_url, autocommit=True),
        embedder=OpenAIEmbedder(model=config.embedding_model),
        floor=config.dense_floor,
    )
    if config.backend is Backend.DENSE:
        return dense
    return HybridPolicyIndex(load_policy_index(policy_dir), dense)
