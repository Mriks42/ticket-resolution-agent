"""What a tool needs in order to run: fixture reads and a place to write."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.paths import resolve_var_dir
from src.tools.fixtures import FixtureBackend
from src.tools.store import WriteStore


@dataclass(frozen=True)
class ToolContext:
    fixtures: FixtureBackend
    store: WriteStore
    var_dir: Path

    @classmethod
    def build(
        cls,
        *,
        var_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
    ) -> ToolContext:
        resolved = resolve_var_dir(var_dir)
        return cls(
            fixtures=FixtureBackend(dataset_dir),
            store=WriteStore(resolved),
            var_dir=resolved,
        )
