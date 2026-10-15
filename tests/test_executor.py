"""The tool executor: read tools through the MCP server, as observations.

Most of these run the real `MCPServer` over the real fixtures, so scopes, the
audit log and the tool handlers are all in play. Two failure cases — a tool
returning malformed output and a tool hanging — cannot be produced by the real
tools on demand, so they use a dispatcher that misbehaves in exactly one way.
"""

from __future__ import annotations

import time

import pytest

from src.agent.executor import MCPToolExecutor
from src.agent.loop import READ_SCOPES
from src.httpjson import RetryPolicy
from src.mcp.server import MCPServer
from src.tools.context import ToolContext
from src.tools.retrieval import RetrievalConfig


@pytest.fixture
def server(tmp_path) -> MCPServer:
    ctx = ToolContext.build(var_dir=tmp_path / "var", retrieval=RetrievalConfig())
    return MCPServer(granted_scopes=READ_SCOPES, context=ctx, var_dir=tmp_path / "var")


class Misbehaving:
    """Answers every tools/call with one fixed kind of wrong."""

    def __init__(self, *, text: str = "{}", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls = 0

    def handle(self, request: dict) -> dict:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"content": [{"type": "text", "text": self.text}], "isError": False},
        }


def no_sleep_retry(attempts: int = 2) -> RetryPolicy:
    return RetryPolicy(max_attempts=attempts, backoff_base=0.0, backoff_cap=0.0)


@pytest.mark.asyncio
async def test_a_read_tool_returns_its_payload_and_is_audited_under_the_run_id(server) -> None:
    executor = MCPToolExecutor(server, timeout=5)
    obs = await executor.call("lookup_customer", {"customer_id": "CUST-1042"}, run_id="run-abc")

    assert obs.status == "ok"
    assert obs.payload["owner_email"] == "r.mehta@brightfold.example"
    [record] = server.audit.records()
    assert record["run_id"] == "run-abc"
    assert record["tool"] == "lookup_customer"


@pytest.mark.asyncio
async def test_a_tool_error_is_an_error_observation_not_an_exception(server) -> None:
    obs = await MCPToolExecutor(server, timeout=5).call(
        "lookup_customer", {"customer_id": "CUST-9999"}, run_id="r"
    )
    assert obs.status == "error"
    assert "CUST-9999" in obs.message


@pytest.mark.asyncio
async def test_bad_arguments_are_an_error_observation(server) -> None:
    obs = await MCPToolExecutor(server, timeout=5).call(
        "lookup_customer", {"customer": "CUST-1042"}, run_id="r"
    )
    assert obs.status == "error"
    assert "bad arguments" in obs.message


@pytest.mark.asyncio
async def test_a_gated_tool_is_refused_by_a_read_only_server_and_audited(server) -> None:
    """Defence in depth: the loop never forwards a gated call, and if a bug
    ever did, the server would refuse it and leave a record saying so."""
    obs = await MCPToolExecutor(server, timeout=5).call(
        "issue_refund", {"ticket_id": "T-1", "amount": 10, "reason": "x"}, run_id="r"
    )
    assert obs.status == "error"
    assert "refund:write" in obs.message
    assert server.audit.records()[-1]["status"] == "refused"
    assert server.context.store.records("refunds") == []


@pytest.mark.asyncio
async def test_malformed_tool_output_is_its_own_observation() -> None:
    """CLAUDE.md failure case: tool returns malformed output."""
    dispatcher = Misbehaving(text="<<not json>>")
    obs = await MCPToolExecutor(dispatcher, timeout=5).call("lookup_customer", {}, run_id="r")
    assert obs.status == "malformed"
    assert obs.payload is None
    assert dispatcher.calls == 1, "malformed output is not retried"


@pytest.mark.asyncio
async def test_output_that_is_json_but_not_an_object_is_malformed() -> None:
    obs = await MCPToolExecutor(Misbehaving(text="[1, 2]"), timeout=5).call(
        "lookup_customer", {}, run_id="r"
    )
    assert obs.status == "malformed"


@pytest.mark.asyncio
async def test_a_hung_tool_times_out_after_retries() -> None:
    """CLAUDE.md failure case: tool times out. Reads are idempotent, so a
    timeout is retried with backoff, then reported."""
    dispatcher = Misbehaving(delay=0.5)
    executor = MCPToolExecutor(dispatcher, timeout=0.05, retry=no_sleep_retry(2))
    obs = await executor.call("lookup_customer", {}, run_id="r")

    assert obs.status == "timeout"
    assert obs.attempts == 2
    assert dispatcher.calls == 2


@pytest.mark.asyncio
async def test_a_timeout_then_success_is_ok() -> None:
    class SlowOnce(Misbehaving):
        def handle(self, request: dict) -> dict:
            self.delay = 0.5 if self.calls == 0 else 0.0
            return super().handle(request)

    executor = MCPToolExecutor(SlowOnce(text='{"plan": "pro"}'), timeout=0.05, retry=no_sleep_retry(3))
    obs = await executor.call("lookup_customer", {}, run_id="r")
    assert obs.status == "ok"
    assert obs.payload == {"plan": "pro"}
    assert obs.attempts == 2


@pytest.mark.asyncio
async def test_what_the_model_sees_names_the_failure() -> None:
    obs = await MCPToolExecutor(Misbehaving(text="nope"), timeout=5).call(
        "lookup_customer", {}, run_id="r"
    )
    seen = obs.for_model()
    assert '"error": "malformed"' in seen
