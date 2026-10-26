"""The run trace: one JSON object per line, append-only, per ADR 0001.

ADR 0001 says the event schema is defined once, here, next to the state
machine. Every line carries `timestamp`, `run_id` and `event`; the rest
depends on the event:

    run_started    ticket_id, customer_id, model, temperature, seed,
                   prompt_version, prompt_sha256, budgets, guardrail_mode,
                   memory, context
    transition     from_state, to_state, reason, step
    llm_call       turn, input_tokens, output_tokens, latency_ms,
                   served_model, finish_reason, tool_calls, cost_usd
    observation    tool, args, status, content (verbatim, what the model
                   saw), latency_ms, attempts
    guardrail      tool, mode (block | detect), violations [{kind, detail, sentence}]
    gate           tool, args, verdict, rule, clause, approver, detail, facts
    escalation_filed  filed_by (model | system), reason, status, message
    memory_written namespace, episode — once per run, at close, memory on only
    run_finished   final_state, outcome, stop_reason, steps, tokens,
                   cost_usd (LLM only), embedding_cost_usd, elapsed_s

The MCP audit log (`src/mcp/audit.py`) is a separate file with a separate
job: it records every tool call the *server* saw. The trace records what the
*loop* decided, including calls it never forwarded — gate decisions,
duplicates, invalid calls. Joined on run_id, the two reconstruct a run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENTS = frozenset(
    {
        "run_started", "transition", "llm_call", "observation", "guardrail", "gate",
        "escalation_filed", "memory_written", "run_finished",
    }
)


class TraceLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_var_dir(cls, var_dir: str | Path) -> TraceLog:
        return cls(Path(var_dir) / "trace" / "trace.jsonl")

    def emit(self, run_id: str, event: str, **fields: Any) -> dict:
        if event not in EVENTS:
            raise ValueError(f"unknown trace event {event!r}")
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def records(self, run_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows if run_id is None else [r for r in rows if r["run_id"] == run_id]
