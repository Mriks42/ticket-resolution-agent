"""Deterministic checks on a generated message, against its spec.

The writer model is told what the message must and must not contain; this is
where that is enforced rather than trusted. A message that fails is
regenerated, and a spec that keeps failing stops the build: nothing is
accepted on the model's word.
"""

from __future__ import annotations

import re

from evals.generate.scenarios import Spec

#: A clause or policy id in a customer message would hand the agent its answer.
_POLICY_ID = re.compile(r"\b(?:POL-)?[A-Z]{3}-\d{3}(?:\.\d+)?\b")
#: Wording from the label side that no customer would write.
_LABEL_WORDS = ("pending_approval", "pending approval", "escalation queue", "support manager approval")

MIN_WORDS, MAX_WORDS = 15, 220


def problems(spec: Spec, body: str) -> list[str]:
    """Everything wrong with `body` for `spec`; empty when it passes."""
    found: list[str] = []
    low = body.lower()
    for options in spec.must_mention:
        if not any(option.lower() in low for option in options):
            found.append(f"missing one of {list(options)}")
    for phrase in spec.must_not_mention:
        if phrase.lower() in low:
            found.append(f"contains {phrase!r}")
    if _POLICY_ID.search(body):
        found.append("contains a policy or clause id")
    for word in _LABEL_WORDS:
        if word in low:
            found.append(f"contains label wording {word!r}")
    words = len(body.split())
    if not MIN_WORDS <= words <= MAX_WORDS:
        found.append(f"{words} words, outside {MIN_WORDS}-{MAX_WORDS}")
    return found
