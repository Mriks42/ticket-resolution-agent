"""What a tool needs in order to run: fixture reads, memory, and a place to write."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.memory import Memory, build_memory
from src.memory.episodic import PgEpisodes
from src.paths import resolve_var_dir
from src.tools.billing import BillingHistory
from src.tools.dense import DensePolicyIndex
from src.tools.embeddings import Embedder
from src.tools.fixtures import FixtureBackend
from src.tools.retrieval import HybridPolicyIndex, RetrievalConfig
from src.tools.store import WriteStore


@dataclass(frozen=True)
class ToolContext:
    fixtures: FixtureBackend
    store: WriteStore
    var_dir: Path
    #: Fresh per context, seeded from the fixtures' past tickets: one eval run
    #: builds one context, so no run sees another's writes (ADR 0008).
    memory: Memory
    #: Refunds received, seeded from the fixtures; per context like memory, but
    #: account state rather than memory (ADR 0011).
    billing: BillingHistory

    @classmethod
    def build(
        cls,
        *,
        var_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        retrieval: RetrievalConfig | None = None,
    ) -> ToolContext:
        resolved = resolve_var_dir(var_dir)
        retrieval = retrieval or RetrievalConfig.from_env()
        fixtures = FixtureBackend(dataset_dir, retrieval)
        return cls(
            fixtures=fixtures,
            store=WriteStore(resolved),
            var_dir=resolved,
            memory=build_memory(retrieval, fixtures.history_seed()),
            billing=BillingHistory.from_customers(fixtures.records),
        )

    def embedders(self) -> list[Embedder]:
        """Every embedder behind these tools, for the cost meter (ADR 0009).
        None under the lexical backend."""
        policy = self.fixtures.policy
        dense = policy.dense if isinstance(policy, HybridPolicyIndex) else policy
        found = [dense.embedder] if isinstance(dense, DensePolicyIndex) else []
        if isinstance(self.memory.episodes, PgEpisodes):
            found.append(self.memory.episodes.embedder)
        return found
