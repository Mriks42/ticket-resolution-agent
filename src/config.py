"""Read `.env` into the environment, stdlib only.

Entry points call `load_dotenv()` once at startup. Library code reads
`os.environ` and never loads files itself, so a test or an eval run controls
its configuration by setting the environment and nothing else. Code that only
needs to *look* at `.env` without changing the process — the test fixtures —
uses `read_dotenv()`.

Values already in the environment win over `.env`, so `FOO=1 python -m ...`
overrides the file the way it would with any other tool.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.paths import REPO_ROOT

DEFAULT_DOTENV = REPO_ROOT / ".env"


def read_dotenv(path: Path = DEFAULT_DOTENV) -> dict[str, str]:
    """`KEY=VALUE` lines, `#` comments, optional matching quotes. A missing
    file is an empty mapping: CI and production configure the environment
    directly."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_dotenv(path: Path = DEFAULT_DOTENV) -> None:
    """Copy `.env` into `os.environ` without overriding what is already set."""
    for key, value in read_dotenv(path).items():
        os.environ.setdefault(key, value)
