"""Where mutable run artifacts live.

Everything the agent writes at runtime — the write-tool store and the audit
log — goes under a single `var/` directory, gitignored, so nothing a run
produces is ever confused with the hand-written fixtures in `evals/dataset/`.

The location is overridable per run, in precedence order:

1. an explicit path passed in code (tests, `evals/run.py`),
2. the `TRA_VAR_DIR` environment variable,
3. `var/` at the repo root.

That ordering is what keeps eval runs and test runs in isolated directories
rather than appending into each other's audit trail.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VAR_DIR_ENV = "TRA_VAR_DIR"
DEFAULT_VAR_DIR = REPO_ROOT / "var"

#: The read-only fixture corpus. Tools read from here and never write to it.
DEFAULT_DATASET_DIR = REPO_ROOT / "evals" / "dataset"


def resolve_var_dir(override: str | Path | None = None) -> Path:
    """Pick the var directory for this run. See module docstring for order."""
    if override is not None:
        return Path(override)
    from_env = os.environ.get(VAR_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_VAR_DIR
