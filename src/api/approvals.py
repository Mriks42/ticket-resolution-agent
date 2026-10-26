"""Approvals: what the gate held, waiting for a human, as an append-only log.

The same format as the trace and the audit log (ADR 0001): one JSON object per
line, appended, never rewritten. Two events:

    held     approval_id, run_id, ticket_id, customer_id, held (the actions)
    decided  approval_id, status (approved | rejected | failed), decided_by,
             note, executed (what each approved action returned)

An approval's state is the fold of its events, so a restart loses nothing
and every decision stays on the record beside what it decided.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ApprovalLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_var_dir(cls, var_dir: str | Path) -> ApprovalLog:
        return cls(Path(var_dir) / "approvals" / "approvals.jsonl")

    def open(self, *, run_id: str, ticket_id: str, customer_id: str, held: list[dict]) -> dict:
        return self._append({
            "event": "held",
            "approval_id": f"apr-{uuid.uuid4().hex[:12]}",
            "run_id": run_id,
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "held": held,
        })

    def decide(
        self, approval_id: str, *, status: str, decided_by: str, note: str | None, executed: list[dict]
    ) -> dict:
        self._append({
            "event": "decided",
            "approval_id": approval_id,
            "status": status,
            "decided_by": decided_by,
            "note": note,
            "executed": executed,
        })
        return self.get(approval_id)

    def get(self, approval_id: str) -> dict | None:
        return self._fold().get(approval_id)

    def pending(self) -> list[dict]:
        """Oldest first."""
        return [a for a in self._fold().values() if a["status"] == "pending"]

    def _fold(self) -> dict[str, dict]:
        approvals: dict[str, dict] = {}
        if not self.path.exists():
            return approvals
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event"] == "held":
                approvals[event["approval_id"]] = {
                    **{k: v for k, v in event.items() if k != "event"},
                    "status": "pending",
                }
            elif event["approval_id"] in approvals:
                approvals[event["approval_id"]].update(
                    {k: v for k, v in event.items() if k not in ("event", "approval_id")},
                    decided_at=event["timestamp"],
                )
        return approvals

    def _append(self, record: dict) -> dict:
        stamped = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        return stamped
