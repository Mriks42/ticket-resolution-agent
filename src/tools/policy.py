"""Parse the policy corpus into individually citable clauses.

CLAUDE.md requires `retrieve_policy` to return clause-level source IDs
(`REF-001.3`) rather than document-level ones (`POL-REF-001`), because the
Session 6 guardrail checks each policy claim against the specific passage
supporting it. A document ID is too coarse for that: `POL-REF-001` covers six
clauses, one of which contradicts a clause in another document, so "backed by
POL-REF-001" is not a check that can fail usefully.

So the unit of retrieval here is the clause, not the file.

Cross-checks are loud, in the style of `evals/loader.py`: a clause whose ID
does not derive from its file's `policy_id`, a duplicate clause ID, or a
document with no clauses at all is a corpus error and raises rather than
quietly indexing something the agent will later cite.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.paths import DEFAULT_DATASET_DIR
from src.tools.lexical import LexicalIndex

DEFAULT_POLICY_DIR = DEFAULT_DATASET_DIR / "policies"

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FRONTMATTER_FIELD = re.compile(r"^([a-z_]+):\s*(.*?)\s*$", re.MULTILINE)
_CLAUSE_MARKER = re.compile(r"^\*\*([A-Z]{3}-\d{3}\.\d+)\*\*", re.MULTILINE)


@dataclass(frozen=True)
class Clause:
    """One citable passage. `clause_id` is what a reply must cite."""

    clause_id: str
    policy_id: str
    title: str
    tiers: tuple[str, ...]
    text: str

    def as_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "policy_id": self.policy_id,
            "title": self.title,
            "tiers": list(self.tiers),
            "text": self.text,
        }


def _parse_frontmatter(path: Path, raw: str) -> dict[str, str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        raise ValueError(f"{path.name}: no YAML frontmatter block")
    return dict(_FRONTMATTER_FIELD.findall(match.group(1)))


def _parse_tiers(value: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in value.strip("[]").split(",") if t.strip())


def parse_policy_file(path: Path) -> list[Clause]:
    """Split one `POL-*.md` into its clauses, validating IDs against the file."""
    raw = path.read_text(encoding="utf-8")
    front = _parse_frontmatter(path, raw)

    policy_id = front.get("policy_id")
    if not policy_id:
        raise ValueError(f"{path.name}: no policy_id in frontmatter")
    if policy_id != path.stem:
        raise ValueError(
            f"{path.name}: declares policy_id {policy_id!r}, "
            f"filename says {path.stem!r}"
        )

    # POL-REF-001 -> clauses must be numbered REF-001.N.
    expected_prefix = policy_id.removeprefix("POL-")

    markers = list(_CLAUSE_MARKER.finditer(raw))
    if not markers:
        raise ValueError(f"{path.name}: no clause markers found")

    title = front.get("title", policy_id)
    tiers = _parse_tiers(front.get("tiers", ""))

    clauses: list[Clause] = []
    for i, marker in enumerate(markers):
        clause_id = marker.group(1)
        if clause_id.rsplit(".", 1)[0] != expected_prefix:
            raise ValueError(
                f"{path.name}: clause {clause_id!r} does not belong to {policy_id}"
            )
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        body = raw[marker.end() : end]
        # Policy prose is hard-wrapped in the source files; a citable passage
        # should read as one paragraph.
        text = " ".join(body.split())
        if not text:
            raise ValueError(f"{path.name}: clause {clause_id!r} has no text")
        clauses.append(
            Clause(
                clause_id=clause_id,
                policy_id=policy_id,
                title=title,
                tiers=tiers,
                text=text,
            )
        )
    return clauses


def load_clauses(policy_dir: Path = DEFAULT_POLICY_DIR) -> list[Clause]:
    """Every clause in the corpus, ordered by document then clause number."""
    clauses: list[Clause] = []
    for path in sorted(policy_dir.glob("POL-*.md")):
        clauses.extend(parse_policy_file(path))
    if not clauses:
        raise ValueError(f"no policy documents found in {policy_dir}")

    counts = Counter(c.clause_id for c in clauses)
    duplicates = sorted(cid for cid, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate clause ids in the corpus: {duplicates}")
    return clauses


class PolicyIndex:
    """Clause-level retrieval over the corpus."""

    def __init__(self, clauses: list[Clause]) -> None:
        self._clauses = {c.clause_id: c for c in clauses}
        # The title is indexed alongside the clause text so a query naming the
        # subject ("refund window") can reach a clause that never repeats it.
        self._index = LexicalIndex(
            {c.clause_id: f"{c.title} {c.text}" for c in clauses}
        )

    def search(self, query: str, limit: int = 5) -> list[tuple[Clause, float]]:
        return [
            (self._clauses[hit.doc_id], hit.score)
            for hit in self._index.search(query, limit)
        ]

    def get(self, clause_id: str) -> Clause | None:
        return self._clauses.get(clause_id)

    def __len__(self) -> int:
        return len(self._clauses)


@lru_cache(maxsize=4)
def load_policy_index(policy_dir: Path = DEFAULT_POLICY_DIR) -> PolicyIndex:
    """Cached per directory — the corpus is static within a run."""
    return PolicyIndex(load_clauses(policy_dir))
