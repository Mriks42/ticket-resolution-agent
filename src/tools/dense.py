"""Clause-level dense retrieval over pgvector.

The same contract as the lexical `PolicyIndex` — `search(query, limit)`
returns `(Clause, score)` pairs, best first, clause-level IDs — so it swaps in
behind `retrieve_policy` without the catalog, the MCP server or the agent loop
noticing. The score is cosine similarity, in [-1, 1].

Three decisions worth stating, each from ADR 0003:

**The similarity floor.** Dense retrieval always has a nearest neighbour, so
without a floor "nothing in the corpus covers this" stops being expressible
and the retrieval guardrail loses the signal it fires on. Hits below `floor`
are dropped. `floor=None` disables it, which exists for calibration only —
the runtime config refuses to build a dense backend without one.

**Every vector is stored, corpus and query alike, keyed by model and the
SHA-256 of the exact text embedded.** A rerun of the eval therefore makes no
API calls and ranks identically, which is what keeps a published number
reproducible after the fact. Only the hash of a query is kept, never its text:
queries are ticket bodies, and the cache is not a place customer prose should
accumulate.

**Search is an exact scan, not an HNSW index.** At ~100 clauses a sequential
scan costs microseconds, and it is exact — an approximate index would add a
second source of ranking variance to an eval that is trying to measure one.
Revisit when the corpus is large enough for the scan to show up in latency.

The schema is applied idempotently on construction; there is no separate
migration step to forget.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from src.tools.embeddings import Embedder
from src.tools.errors import ToolError
from src.tools.policy import Clause, indexed_text

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embedding (
    model        text        NOT NULL,
    text_sha256  text        NOT NULL,
    dims         integer     NOT NULL,
    embedding    vector      NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, text_sha256),
    CHECK (vector_dims(embedding) = dims)
);

CREATE TABLE IF NOT EXISTS policy_clause (
    model        text NOT NULL,
    clause_id    text NOT NULL,
    policy_id    text NOT NULL,
    text_sha256  text NOT NULL,
    PRIMARY KEY (model, clause_id),
    FOREIGN KEY (model, text_sha256) REFERENCES embedding (model, text_sha256)
);
"""

_SEARCH = """
SELECT pc.clause_id, 1 - (e.embedding <=> q.embedding) AS similarity
FROM policy_clause AS pc
JOIN embedding AS e
  ON e.model = pc.model AND e.text_sha256 = pc.text_sha256
CROSS JOIN (
    SELECT embedding FROM embedding
    WHERE model = %(model)s AND text_sha256 = %(query)s
) AS q
WHERE pc.model = %(model)s AND pc.clause_id = ANY(%(ids)s)
ORDER BY e.embedding <=> q.embedding, pc.clause_id
LIMIT %(limit)s
"""


class DenseIndexError(ToolError):
    """The vector store could not answer a search."""


@dataclass
class DenseStats:
    """Counters for the eval report: what a run cost, and what came from cache."""

    clauses_embedded: int = 0
    query_embeds: int = 0
    query_cache_hits: int = 0


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def embed_missing(conn: psycopg.Connection, embedder: Embedder, texts: Sequence[str]) -> int:
    """Embed and store whichever of `texts` have no vector under
    `embedder.model` yet, and return how many that was. Rows are keyed by the
    text's hash; the text itself is not stored. Shared by every table that
    joins to `embedding`, so a text is embedded once whoever asks first."""
    by_hash = {text_sha256(t): t for t in texts}
    have = {
        row[0]
        for row in conn.execute(
            "SELECT text_sha256 FROM embedding WHERE model = %s AND text_sha256 = ANY(%s)",
            (embedder.model, list(by_hash)),
        )
    }
    todo = sorted(h for h in by_hash if h not in have)
    if not todo:
        return 0
    vectors = embedder.embed([by_hash[h] for h in todo])
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO embedding (model, text_sha256, dims, embedding) "
            "VALUES (%s, %s, %s, %s::vector) "
            "ON CONFLICT (model, text_sha256) DO NOTHING",
            [
                (embedder.model, h, embedder.dims, _vector_literal(v))
                for h, v in zip(todo, vectors, strict=True)
            ],
        )
    return len(todo)


class DensePolicyIndex:
    """Cosine-similarity search over clause embeddings stored in Postgres."""

    backend = "dense"

    def __init__(
        self,
        clauses: list[Clause],
        *,
        conn: psycopg.Connection,
        embedder: Embedder,
        floor: float | None,
    ) -> None:
        self._clauses = {c.clause_id: c for c in clauses}
        self._conn = conn
        self._embedder = embedder
        self.model = embedder.model
        self.floor = floor
        self.stats = DenseStats()

        conn.execute(SCHEMA)
        self._sync(clauses)

    # --- contract shared with PolicyIndex ---------------------------------

    def search(self, query: str, limit: int = 5) -> list[tuple[Clause, float]]:
        if not query.strip():
            return []
        try:
            query_hash = self._ensure_query_embedded(query)
            rows = self._conn.execute(
                _SEARCH,
                {
                    "model": self.model,
                    "query": query_hash,
                    "ids": list(self._clauses),
                    "limit": limit,
                },
            ).fetchall()
        except psycopg.Error as exc:
            raise DenseIndexError(f"policy search failed: {exc}") from exc

        hits = [(self._clauses[clause_id], float(sim)) for clause_id, sim in rows]
        if self.floor is not None:
            hits = [(c, s) for c, s in hits if s >= self.floor]
        return hits

    def get(self, clause_id: str) -> Clause | None:
        return self._clauses.get(clause_id)

    def __len__(self) -> int:
        return len(self._clauses)

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # --- storage -----------------------------------------------------------

    def _sync(self, clauses: list[Clause]) -> None:
        """Make `policy_clause` match `clauses` for this model, embedding only
        text that has never been embedded under it."""
        self.stats.clauses_embedded += embed_missing(
            self._conn, self._embedder, [indexed_text(c) for c in clauses]
        )

        with self._conn.transaction():
            self._conn.execute(
                "DELETE FROM policy_clause WHERE model = %s AND NOT (clause_id = ANY(%s))",
                (self.model, [c.clause_id for c in clauses]),
            )
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO policy_clause (model, clause_id, policy_id, text_sha256) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (model, clause_id) DO UPDATE "
                    "SET policy_id = EXCLUDED.policy_id, text_sha256 = EXCLUDED.text_sha256",
                    [
                        (self.model, c.clause_id, c.policy_id, text_sha256(indexed_text(c)))
                        for c in clauses
                    ],
                )

    def _ensure_query_embedded(self, query: str) -> str:
        if embed_missing(self._conn, self._embedder, [query]):
            self.stats.query_embeds += 1
        else:
            self.stats.query_cache_hits += 1
        return text_sha256(query)
