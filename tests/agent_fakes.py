"""Fakes for driving the agent loop: a scripted LLM and a misbehaving server.

The LLM is the only thing faked in the loop tests. Tools run for real.
"""

from __future__ import annotations

import json
import time
from itertools import count

from src.agent.llm import Completion, ToolCallRequest, ToolResult, Usage

_ids = count(1)


def completion(name, args, *, tokens=(100, 20), also=()) -> Completion:
    """One model turn calling `name`. `args` may be a dict or a raw string."""

    def request(n, a):
        raw = a if isinstance(a, str) else json.dumps(a)
        return ToolCallRequest(call_id=f"call_{next(_ids)}", name=n, arguments=raw)

    calls = (request(name, args), *(request(n, a) for n, a in also))
    return Completion(
        text=None,
        tool_calls=calls,
        usage=Usage(*tokens),
        served_model="scripted-2026-10-14",
        finish_reason="tool_calls",
    )


def text_only(text="Let me think about this.") -> Completion:
    return Completion(text=text, tool_calls=(), usage=Usage(100, 20), served_model="scripted")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ScriptedLLM:
    """Plays back completions in order and records what it was shown."""

    model = "scripted"

    def __init__(self, *steps, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        self.steps = list(steps)
        self.seen: list[list] = []
        self.tools_offered: list[str] = []
        self.clock = clock
        self.advance = advance

    async def complete(self, messages, tools):
        self.seen.append(list(messages))
        self.tools_offered = [t.name for t in tools]
        if self.clock is not None:
            self.clock.now += self.advance
        if not self.steps:
            raise AssertionError("scripted LLM ran out of steps")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return await step()
        return step

    def last_observation(self, turn: int) -> str:
        """The tool result the model saw at the start of `turn` (0-based)."""
        results = [m for m in self.seen[turn] if isinstance(m, ToolResult)]
        return results[-1].content


class Misbehaving:
    def __init__(self, *, text: str = "{}", delay: float = 0.0) -> None:
        self.text, self.delay = text, delay

    def handle(self, request: dict) -> dict:
        if self.delay:
            time.sleep(self.delay)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"content": [{"type": "text", "text": self.text}], "isError": False},
        }
