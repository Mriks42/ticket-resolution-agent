"""The memory layer (ADR 0008): episodic and semantic, kept separate.

`Memory` is what a `ToolContext` holds. The read tools consult it —
`search_past_tickets`, and `past_tickets` and `memory` in `lookup_customer` —
and the loop writes to it once per run, at close, while memory is on.
Whether memory is on is the loop's setting (`AgentConfig.memory`), not this
module's: off, the loop neither shows the agent what is here nor writes to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

from src.memory.close import Close
from src.memory.episodic import Episode, EpisodicStore, InProcessEpisodes, PgEpisodes
from src.memory.semantic import FactStore, InProcessFacts, PgFacts, fold
from src.tools.embeddings import OpenAIEmbedder
from src.tools.retrieval import Backend, RetrievalConfig


@dataclass(frozen=True)
class Memory:
    episodes: EpisodicStore
    facts: FactStore

    def __post_init__(self) -> None:
        if self.episodes.namespace != self.facts.namespace:
            raise ValueError(
                f"both layers must share a namespace, got {self.episodes.namespace!r} "
                f"and {self.facts.namespace!r}"
            )

    @property
    def namespace(self) -> str:
        return self.episodes.namespace

    def remember(self, close: Close) -> dict:
        """Write what a finished run leaves behind to both layers, and return
        it for the trace."""
        episode = Episode.from_close(close)
        facts = fold(self.facts.get(close.customer_id), close)
        self.episodes.add(episode)
        self.facts.put(close.customer_id, facts)
        return {"namespace": self.namespace, "episode": episode.as_dict(), "facts": facts}

    def describe(self) -> dict:
        return {
            "episodic": self.episodes.backend,
            "semantic": self.facts.backend,
            "namespace": self.namespace,
        }


def build_memory(
    config: RetrievalConfig, seed: Iterable[Episode], *, namespace: str | None = None
) -> Memory:
    """Memory for one eval run, or one server: a fresh namespace unless one is
    given, seeded with `seed`. The backend follows `config.backend`."""
    namespace = namespace or f"mem-{uuid4().hex[:12]}"
    if config.backend is Backend.LEXICAL:
        return Memory(InProcessEpisodes(seed, namespace=namespace), InProcessFacts(namespace=namespace))

    import psycopg

    conn = psycopg.connect(config.database_url, autocommit=True)
    return Memory(
        PgEpisodes(
            seed,
            conn=conn,
            embedder=OpenAIEmbedder(model=config.embedding_model),
            namespace=namespace,
        ),
        PgFacts(conn=conn, namespace=namespace),
    )
