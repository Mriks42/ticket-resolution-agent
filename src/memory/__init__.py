"""The memory layer (ADR 0008).

`Memory` is what a `ToolContext` holds. The read tools consult it —
`search_past_tickets` and the `past_tickets` in `lookup_customer` — and the
loop writes to it once per run, at close, while memory is on. Whether memory
is on is the loop's setting (`AgentConfig.memory`), not this module's: off,
the loop neither shows the agent what is here nor writes to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import uuid4

from src.memory.close import Close
from src.memory.episodic import Episode, EpisodicStore, InProcessEpisodes, PgEpisodes
from src.tools.embeddings import OpenAIEmbedder
from src.tools.retrieval import Backend, RetrievalConfig


@dataclass(frozen=True)
class Memory:
    episodes: EpisodicStore

    @property
    def namespace(self) -> str:
        return self.episodes.namespace

    def remember(self, close: Close) -> dict:
        """Write what a finished run leaves behind, and return it for the trace."""
        episode = Episode.from_close(close)
        self.episodes.add(episode)
        return {"namespace": self.namespace, "episode": episode.as_dict()}

    def describe(self) -> dict:
        return {"episodic": self.episodes.backend, "namespace": self.namespace}


def build_memory(
    config: RetrievalConfig, seed: Iterable[Episode], *, namespace: str | None = None
) -> Memory:
    """Memory for one eval run, or one server: a fresh namespace unless one is
    given, seeded with `seed`. The backend follows `config.backend`."""
    namespace = namespace or f"mem-{uuid4().hex[:12]}"
    if config.backend is Backend.LEXICAL:
        return Memory(InProcessEpisodes(seed, namespace=namespace))

    import psycopg

    return Memory(
        PgEpisodes(
            seed,
            conn=psycopg.connect(config.database_url, autocommit=True),
            embedder=OpenAIEmbedder(model=config.embedding_model),
            namespace=namespace,
        )
    )
