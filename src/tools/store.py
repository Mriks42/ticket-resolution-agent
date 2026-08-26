"""Append-only JSON Lines store for what the write tools actually did.

One file per record kind under `<var_dir>/store/`. Same format as the audit log
per ADR 0001: one JSON object per line, appended, never rewritten.

This is the *effect* log — the refund that was issued, the reply that was sent.
The audit log next to it records that the call happened at all, including calls
that were refused and therefore have no effect here. Keeping them separate
means a refused write leaves a trace without leaving a record that looks like
an action was taken.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

#: Fixed set, so a typo cannot quietly create a fourth store file that nothing
#: ever reads back.
STORE_KINDS = ("refunds", "replies", "escalations")


class WriteStore:
    def __init__(self, var_dir: str | Path) -> None:
        self.dir = Path(var_dir) / "store"

    def path(self, kind: str) -> Path:
        self._check(kind)
        return self.dir / f"{kind}.jsonl"

    def append(self, kind: str, record: dict) -> dict:
        """Write one record and return it, stamped with an id and a timestamp."""
        self._check(kind)
        stamped = {
            "id": f"{kind[:3]}-{uuid4().hex[:12]}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        path = self.path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        return stamped

    def records(self, kind: str) -> list[dict]:
        path = self.path(kind)
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _check(kind: str) -> None:
        if kind not in STORE_KINDS:
            raise ValueError(
                f"unknown store kind {kind!r}; expected one of {STORE_KINDS}"
            )
