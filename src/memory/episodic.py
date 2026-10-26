"""Episodic memory: a customer's past tickets and how they ended.

One store backs both `search_past_tickets` (ranked by similarity to the
query) and `past_tickets` in `lookup_customer` (the whole history, oldest
first), so the two cannot disagree. It starts from a seed — in an eval, the
fixtures' `past_tickets`, which are that world's prior history — and grows by
one episode per closed run while memory is on.

Two backends, picked by the same setting as policy retrieval (ADR 0008):

- **InProcessEpisodes** for `lexical`: the TF-IDF index the search always
  used. No dependencies, so tests and CI run without Postgres.
- **PgEpisodes** for `dense` and `hybrid`: cosine similarity over pgvector,
  joined to the embedding cache of ADR 0004, so text already embedded by any
  earlier run costs nothing.

**No similarity floor.** An empty result tells the agent the customer has
never written in. A floor would make real history vanish whenever the query
was phrased differently, so a search scoped to one customer returns that
customer's history ranked, up to the limit. The in-process backend lists the
episodes that share no word with the query after the ones that do, at score
0, so empty means the same thing under both.

**Namespaces.** Every store belongs to one namespace, and an eval run gets a
fresh one. Writes accumulate within the run and never reach another, so a
rerun starts from the same state.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

import psycopg

from src.memory.close import Close
from src.tools.dense import SCHEMA as EMBEDDING_SCHEMA
from src.tools.dense import embed_missing, text_sha256
from src.tools.embeddings import Embedder
from src.tools.errors import ToolError
from src.tools.lexical import LexicalIndex

OUTCOME_TEXT = {
    "RESOLVE": "Resolved by the agent; a reply was sent.",
    "PENDING_APPROVAL": "Waiting on an approver.",
    "ESCALATE": "Escalated to a human.",
    "FAILED": "Escalated to a human; the agent could not complete the run.",
}
REFUND_TEXT = {"issued": "issued", "held": "held for approval"}


@dataclass(frozen=True)
class Episode:
    customer_id: str
    ticket_id: str
    opened: str
    summary: str
    resolution: str

    @property
    def text(self) -> str:
        """What is indexed and embedded."""
        return f"{self.summary} {self.resolution}"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_close(cls, close: Close) -> Episode:
        return cls(close.customer_id, close.ticket_id, close.date, close.body, resolution_text(close))


def resolution_text(close: Close) -> str:
    """The outcome of a run as a fixed template: terminal state, refunds,
    clause ids. No reply text and no escalation reason (ADR 0008)."""
    if close.final_state not in OUTCOME_TEXT:
        raise ValueError(f"{close.final_state!r} is not a terminal state")
    parts = [OUTCOME_TEXT[close.final_state]]
    parts += [f"Refund of ${r.amount:,.2f} {REFUND_TEXT[r.status]}." for r in close.refunds]
    if close.clauses:
        parts.append(f"Clauses cited: {', '.join(close.clauses)}.")
    return " ".join(parts)


class EpisodicStoreError(ToolError):
    """The episode store could not answer or could not write."""


class EpisodicStore(Protocol):
    backend: str
    namespace: str

    def history(self, customer_id: str) -> list[Episode]: ...

    def search(
        self, query: str, customer_id: str | None, limit: int
    ) -> list[tuple[Episode, float]]: ...

    def add(self, episode: Episode) -> None: ...


class InProcessEpisodes:
    backend = "in-process"

    def __init__(self, seed: Iterable[Episode], *, namespace: str) -> None:
        self.namespace = namespace
        self._episodes: list[Episode] = []
        for episode in seed:
            self._put(episode)
        self._reindex()

    def history(self, customer_id: str) -> list[Episode]:
        # sorted() is stable, so episodes opened the same day stay in the
        # order they were added.
        return sorted((e for e in self._episodes if e.customer_id == customer_id), key=lambda e: e.opened)

    def search(
        self, query: str, customer_id: str | None = None, limit: int = 5
    ) -> list[tuple[Episode, float]]:
        ranked = [
            (self._by_key[hit.doc_id], hit.score)
            for hit in self._index.search(query)
            if customer_id is None or self._by_key[hit.doc_id].customer_id == customer_id
        ]
        if customer_id is not None:
            matched = {e.ticket_id for e, _ in ranked}
            ranked += [(e, 0.0) for e in self.history(customer_id) if e.ticket_id not in matched]
        return ranked[:limit]

    def add(self, episode: Episode) -> None:
        self._put(episode)
        self._reindex()

    def _put(self, episode: Episode) -> None:
        key = (episode.customer_id, episode.ticket_id)
        self._episodes = [e for e in self._episodes if (e.customer_id, e.ticket_id) != key]
        self._episodes.append(episode)

    def _reindex(self) -> None:
        self._by_key = {f"{e.customer_id}/{e.ticket_id}": e for e in self._episodes}
        self._index = LexicalIndex({key: e.text for key, e in self._by_key.items()})


SCHEMA = """
CREATE TABLE IF NOT EXISTS episode (
    namespace    text      NOT NULL,
    customer_id  text      NOT NULL,
    ticket_id    text      NOT NULL,
    opened       text      NOT NULL,
    summary      text      NOT NULL,
    resolution   text      NOT NULL,
    model        text      NOT NULL,
    text_sha256  text      NOT NULL,
    -- Insertion order, the tie-break for episodes opened the same day.
    seq          bigserial,
    PRIMARY KEY (namespace, customer_id, ticket_id),
    FOREIGN KEY (model, text_sha256) REFERENCES embedding (model, text_sha256)
);
"""

_COLUMNS = "ep.customer_id, ep.ticket_id, ep.opened, ep.summary, ep.resolution"

_SEARCH = f"""
SELECT {_COLUMNS}, 1 - (e.embedding <=> q.embedding) AS similarity
FROM episode AS ep
JOIN embedding AS e
  ON e.model = ep.model AND e.text_sha256 = ep.text_sha256
CROSS JOIN (
    SELECT embedding FROM embedding
    WHERE model = %(model)s AND text_sha256 = %(query)s
) AS q
WHERE ep.namespace = %(namespace)s AND ep.model = %(model)s
  AND (%(customer)s::text IS NULL OR ep.customer_id = %(customer)s)
ORDER BY e.embedding <=> q.embedding, ep.ticket_id
LIMIT %(limit)s
"""

_HISTORY = f"""
SELECT {_COLUMNS} FROM episode AS ep
WHERE ep.namespace = %s AND ep.customer_id = %s
ORDER BY ep.opened, ep.seq
"""

_UPSERT = """
INSERT INTO episode
    (namespace, customer_id, ticket_id, opened, summary, resolution, model, text_sha256)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (namespace, customer_id, ticket_id) DO UPDATE SET
    opened = EXCLUDED.opened, summary = EXCLUDED.summary,
    resolution = EXCLUDED.resolution, model = EXCLUDED.model,
    text_sha256 = EXCLUDED.text_sha256, seq = DEFAULT
"""


class PgEpisodes:
    backend = "pgvector"

    def __init__(
        self,
        seed: Iterable[Episode],
        *,
        conn: psycopg.Connection,
        embedder: Embedder,
        namespace: str,
    ) -> None:
        self.namespace = namespace
        self.model = embedder.model
        self._conn = conn
        self._embedder = embedder
        conn.execute(EMBEDDING_SCHEMA)
        conn.execute(SCHEMA)
        self._write(list(seed))

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def history(self, customer_id: str) -> list[Episode]:
        try:
            rows = self._conn.execute(_HISTORY, (self.namespace, customer_id)).fetchall()
        except psycopg.Error as exc:
            raise EpisodicStoreError(f"history read failed: {exc}") from exc
        return [Episode(*row) for row in rows]

    def search(
        self, query: str, customer_id: str | None = None, limit: int = 5
    ) -> list[tuple[Episode, float]]:
        if not query.strip():
            # Nothing to rank by; the history itself is the answer.
            return [(e, 0.0) for e in self.history(customer_id)][:limit] if customer_id else []
        try:
            embed_missing(self._conn, self._embedder, [query])
            rows = self._conn.execute(
                _SEARCH,
                {
                    "model": self.model,
                    "query": text_sha256(query),
                    "namespace": self.namespace,
                    "customer": customer_id,
                    "limit": limit,
                },
            ).fetchall()
        except psycopg.Error as exc:
            raise EpisodicStoreError(f"past-ticket search failed: {exc}") from exc
        return [(Episode(*row[:5]), float(row[5])) for row in rows]

    def add(self, episode: Episode) -> None:
        self._write([episode])

    def _write(self, episodes: list[Episode]) -> None:
        if not episodes:
            return
        try:
            embed_missing(self._conn, self._embedder, [e.text for e in episodes])
            with self._conn.transaction(), self._conn.cursor() as cur:
                cur.executemany(
                    _UPSERT,
                    [
                        (
                            self.namespace, e.customer_id, e.ticket_id, e.opened,
                            e.summary, e.resolution, self.model, text_sha256(e.text),
                        )
                        for e in episodes
                    ],
                )
        except psycopg.Error as exc:
            raise EpisodicStoreError(f"episode write failed: {exc}") from exc
