"""Run read tools through the MCP server and report what happened.

The loop never calls a tool handler directly. It sends a JSON-RPC
`tools/call` to an `MCPServer` granted read scopes only, so the permission
check and the audit record happen for every call the agent makes, exactly as
they would over stdio. The run_id travels in `_meta`, which is where the
server's audit log reads it from.

Everything comes back as an `Observation`, never an exception: per CLAUDE.md a
failed tool narrows the agent's options and does not crash the run. Four
outcomes, each a distinct status the model and the trace can see:

- `ok`         a JSON object, as the tool returned it
- `error`      the tool ran and said no, or the server refused the call
- `malformed`  the tool answered with something that is not a JSON object
- `timeout`    no answer within `timeout`, after retries

Only timeouts are retried. Every tool reachable here is a read, so repeating
one is safe; an error or a malformed answer would come back the same way.

A timed-out call's worker thread is abandoned, not killed — Python cannot
kill a thread. It finishes in the background and its result is discarded.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from src.httpjson import RetryPolicy

Status = Literal["ok", "error", "malformed", "timeout"]


@dataclass(frozen=True)
class Observation:
    status: Status
    payload: dict | None = None
    message: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1

    def for_model(self) -> str:
        """What the model is shown, as the tool message content."""
        if self.status == "ok":
            return json.dumps(self.payload, ensure_ascii=False)
        return json.dumps({"error": self.status, "message": self.message}, ensure_ascii=False)


class Dispatcher(Protocol):
    """Anything that answers JSON-RPC requests. `MCPServer` is one."""

    def handle(self, request: dict) -> dict | None: ...


class ToolExecutor(Protocol):
    async def call(self, name: str, arguments: dict, *, run_id: str) -> Observation: ...


class MCPToolExecutor:
    def __init__(
        self,
        server: Dispatcher,
        *,
        timeout: float = 20.0,
        retry: RetryPolicy = RetryPolicy(max_attempts=2, backoff_base=0.5, backoff_cap=4.0),
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._server = server
        self._timeout = timeout
        self._retry = retry
        self._sleep = sleep
        self._ids = itertools.count(1)

    async def call(self, name: str, arguments: dict, *, run_id: str) -> Observation:
        started = time.perf_counter()
        for attempt in range(1, self._retry.max_attempts + 1):
            if attempt > 1:
                await self._sleep(self._retry.delay(attempt - 1))
            request = {
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments, "_meta": {"run_id": run_id}},
            }
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._server.handle, request), self._timeout
                )
            except TimeoutError:
                continue
            return _interpret(response, _ms(started), attempt)

        return Observation(
            "timeout",
            message=(
                f"{name} did not answer within {self._timeout:g}s "
                f"({self._retry.max_attempts} attempts)"
            ),
            latency_ms=_ms(started),
            attempts=self._retry.max_attempts,
        )


def _interpret(response: dict | None, latency_ms: float, attempts: int) -> Observation:
    def obs(status: Status, **kw) -> Observation:
        return Observation(status, latency_ms=latency_ms, attempts=attempts, **kw)

    if not isinstance(response, dict):
        return obs("malformed", message="the server sent no response")
    if "error" in response:
        error = response["error"] if isinstance(response["error"], dict) else {}
        return obs("error", message=str(error.get("message", "unknown error")))
    try:
        result = response["result"]
        text = result["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return obs("malformed", message="the response carried no text content")
    if result.get("isError"):
        return obs("error", message=str(text))
    try:
        payload = json.loads(text)
    except ValueError:
        return obs("malformed", message="the tool's output was not valid JSON")
    if not isinstance(payload, dict):
        return obs("malformed", message="the tool's output was JSON but not an object")
    return obs("ok", payload=payload)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
