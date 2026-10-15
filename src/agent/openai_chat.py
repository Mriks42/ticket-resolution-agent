"""`LLMClient` over OpenAI's chat completions API, stdlib only.

One POST per planning step, through `src/httpjson.py` so it shares the
embedder's retry policy and never echoes the key. The blocking call runs in a
worker thread so the loop's wall-clock budget can cut it off.

Two request settings carry the design:

- `tool_choice="required"` with `parallel_tool_calls=false`: every turn is
  exactly one tool call. That is ReAct one step at a time, as CLAUDE.md
  specifies, enforced by the provider rather than requested in prose.
- `temperature=0` and a fixed `seed` for eval runs. OpenAI documents the seed
  as best-effort determinism, not a guarantee, so runs are *more* repeatable,
  not identical. The served model snapshot is returned on every completion so
  the trace records which model actually answered.

The adapter does not parse tool-call arguments. See `src/agent/llm.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from src.agent.llm import (
    Assistant,
    Completion,
    FunctionSchema,
    LLMError,
    Message,
    System,
    ToolCallRequest,
    ToolResult,
    Usage,
    User,
)
from src.httpjson import HTTPRejected, HTTPRetriesExhausted, RetryPolicy, post_json

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
#: Same seed as the eval bootstrap, so one constant pins a run.
DEFAULT_SEED = 20260824


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_CHAT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        seed: int | None = DEFAULT_SEED,
        timeout: float = 60.0,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set; the agent loop cannot plan without "
                "a model. Add it to .env or the environment."
            )
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._key = key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._retry = RetryPolicy(max_attempts, backoff_base, backoff_cap)
        self._sleep = sleep

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[FunctionSchema]
    ) -> Completion:
        payload = self._payload(messages, tools)
        try:
            raw = await asyncio.to_thread(
                post_json,
                self._url,
                payload,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=self._timeout,
                policy=self._retry,
                sleep=self._sleep,
                redact=(self._key,),
            )
        except HTTPRejected as exc:
            raise LLMError(f"chat request rejected, {exc}") from None
        except HTTPRetriesExhausted as exc:
            raise LLMError(f"chat request {exc}") from None
        return _parse(raw)

    def _payload(self, messages: Sequence[Message], tools: Sequence[FunctionSchema]) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_wire_message(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
        return payload


def _wire_message(message: Message) -> dict:
    if isinstance(message, System):
        return {"role": "system", "content": message.content}
    if isinstance(message, User):
        return {"role": "user", "content": message.content}
    if isinstance(message, Assistant):
        wire: dict[str, Any] = {"role": "assistant", "content": message.text}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": c.call_id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in message.tool_calls
            ]
        return wire
    if isinstance(message, ToolResult):
        return {"role": "tool", "tool_call_id": message.call_id, "content": message.content}
    raise TypeError(f"not a message: {message!r}")


def _parse(raw: bytes) -> Completion:
    try:
        body = json.loads(raw)
        choice = body["choices"][0]
        message = choice["message"]
        usage = body["usage"]
        input_tokens, output_tokens = usage["prompt_tokens"], usage["completion_tokens"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("malformed chat response: missing choices, message or usage") from None
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        raise LLMError("malformed chat response: usage is not token counts")

    calls = []
    for wire in message.get("tool_calls") or []:
        try:
            call_id, function = wire["id"], wire["function"]
            name, arguments = function["name"], function["arguments"]
        except (KeyError, TypeError):
            raise LLMError("malformed chat response: incomplete tool call") from None
        if not all(isinstance(v, str) for v in (call_id, name, arguments)):
            raise LLMError("malformed chat response: tool call fields are not strings")
        calls.append(ToolCallRequest(call_id=call_id, name=name, arguments=arguments))

    text = message.get("content")
    return Completion(
        text=text if isinstance(text, str) else None,
        tool_calls=tuple(calls),
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        served_model=str(body.get("model", "")),
        finish_reason=choice.get("finish_reason"),
    )
