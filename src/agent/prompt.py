"""The versioned system prompt.

The prompt is part of the agent's configuration in the same way the model is:
change a sentence and the eval numbers can move. So it lives in a file named
by version, and every run records the version and the SHA-256 of the exact
text it used. Two runs are comparable only if both match.

The prompt holds behaviour rules and no policy content. Policy reaches the
model only through `retrieve_policy`, which is what lets the guardrail check
every policy claim against a retrieved passage.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
#: v2 (session 6) changes only the last paragraph: gated actions now pass
#: through the HITL gate instead of being held unconditionally. v1 is kept so
#: runs recorded under it stay reproducible.
DEFAULT_PROMPT_VERSION = "system_v2"


@dataclass(frozen=True)
class Prompt:
    version: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_prompt(version: str = DEFAULT_PROMPT_VERSION) -> Prompt:
    path = PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt {version!r} at {path}")
    return Prompt(version=version, text=path.read_text(encoding="utf-8").strip())
