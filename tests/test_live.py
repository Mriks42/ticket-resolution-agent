"""One real ticket through the real model. Opt-in, and it costs money.

    TRA_LIVE=1 uv run pytest tests/test_live.py -q

Skipped by default. Everything else in the suite runs against a scripted
LLM; this is the one test that catches the provider changing its wire
format, its tool-call behaviour or its usage reporting under us. It asserts
that the loop *works* against the real model — a terminal state reached
without FAILED, real usage recorded, cost inside a small cap — not that the
model answers correctly. Correctness is `evals/run.py`'s question.

Tools run on the lexical backend so the test needs no database.
"""

from __future__ import annotations

import os

import pytest

from evals.loader import load_tickets
from src.agent.config import AgentConfig
from src.agent.executor import MCPToolExecutor
from src.agent.loop import READ_SCOPES, WRITE_SCOPES, AgentLoop
from src.agent.openai_chat import OpenAIChatClient
from src.agent.pricing import SpendGuard
from src.agent.trace import TraceLog
from src.config import read_dotenv
from src.mcp.server import MCPServer
from src.tools.context import ToolContext
from src.tools.retrieval import RetrievalConfig

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("TRA_LIVE") != "1", reason="live model test; set TRA_LIVE=1"),
]


@pytest.mark.asyncio
async def test_one_ticket_end_to_end_against_the_real_model(tmp_path) -> None:
    key = os.environ.get("OPENAI_API_KEY") or read_dotenv().get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")

    ticket = next(t for t in load_tickets() if t.ticket_id == "T-001")
    ctx = ToolContext.build(var_dir=tmp_path / "var", retrieval=RetrievalConfig())
    server = MCPServer(granted_scopes=READ_SCOPES, context=ctx, var_dir=tmp_path / "var")
    writer = MCPServer(granted_scopes=WRITE_SCOPES, context=ctx, var_dir=tmp_path / "var")
    trace = TraceLog(tmp_path / "trace.jsonl")
    config = AgentConfig()
    guard = SpendGuard.for_model(config.model, cap_usd=0.05)

    loop = AgentLoop(
        llm=OpenAIChatClient(model=config.model, api_key=key),
        executor=MCPToolExecutor(server, timeout=config.tool_timeout),
        writer=MCPToolExecutor(writer, timeout=config.tool_timeout),
        trace=trace,
        config=config,
        spend=guard,
    )
    result = await loop.run(ticket)

    assert result.final_state in {"RESOLVE", "ESCALATE", "PENDING_APPROVAL"}, result.stop_reason
    assert result.tool_calls, "the model called no tool at all"
    calls = [e for e in trace.records(result.run_id) if e["event"] == "llm_call"]
    assert calls and all(c["input_tokens"] > 0 for c in calls)
    assert calls[0]["served_model"].startswith("gpt-4o-mini")
    assert 0 < guard.spent_usd < 0.05
    print(
        f"\nlive: {result.final_state}, {len(result.tool_calls)} calls "
        f"({', '.join(c.tool.value for c in result.tool_calls)}), "
        f"{len(calls)} LLM turns, ${guard.spent_usd:.5f}, model {calls[0]['served_model']}"
    )
