"""Append-only audit log, one JSON object per line, per ADR 0001.

CLAUDE.md requires every tool call to write a record carrying run_id, tool,
args, caller scope, result status, latency and timestamp. "Every tool call"
includes the ones that never ran: a call refused for want of a scope is
recorded with `status="refused"`, because a permission system that leaves no
evidence when it says no cannot be audited for whether it ever said no.

There is deliberately no update or delete. The only write path is `record()`,
which opens the file in append mode and closes it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Status = Literal["ok", "refused", "error"]

#: Fields every record carries, whatever the outcome.
REQUIRED_FIELDS = (
    "timestamp",
    "run_id",
    "tool",
    "args",
    "caller_scopes",
    "required_scope",
    "status",
    "latency_ms",
)


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_var_dir(cls, var_dir: str | Path) -> AuditLog:
        return cls(Path(var_dir) / "audit" / "audit.jsonl")

    def record(
        self,
        *,
        run_id: str,
        tool: str,
        args: dict[str, Any],
        caller_scopes: list[str],
        required_scope: str | None,
        status: Status,
        latency_ms: float,
        detail: str | None = None,
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "tool": tool,
            "args": args,
            "caller_scopes": caller_scopes,
            "required_scope": required_scope,
            "status": status,
            "latency_ms": latency_ms,
        }
        if detail is not None:
            entry["detail"] = detail

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def records(self) -> list[dict]:
        """Read the log back. Used by tests and by `evals/run.py`."""
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
