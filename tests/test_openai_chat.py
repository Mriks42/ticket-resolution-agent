"""The OpenAI chat adapter, against a scripted local HTTP server.

Pins the wire format both ways: what the adapter sends (one tool call per turn,
temperature and seed for reproducibility) and what it accepts back. No network,
no key, nothing in the eval path.
"""

from __future__ import annotations

import json

import pytest

from src.agent.llm import (
    Assistant,
    FunctionSchema,
    LLMError,
    System,
    ToolCallRequest,
    ToolResult,
    User,
)
from src.agent.openai_chat import OpenAIChatClient

API_KEY = "sk-test-not-a-real-key"

LOOKUP = FunctionSchema(
    name="lookup_customer",
    description="Fetch the account record.",
    parameters={
        "type": "object",
        "properties": {"customer_id": {"type": "string"}},
        "required": ["customer_id"],
    },
)


def reply(
    *,
    calls: list[tuple[str, str, str]] = (),
    text: str | None = None,
    prompt_tokens: int = 120,
    completion_tokens: int = 15,
    model: str = "gpt-4o-mini-2024-07-18",
    finish_reason: str = "tool_calls",
) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in calls
        ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@pytest.fixture
def sleeps() -> list[float]:
    return []


def make(api, sleeps, **overrides) -> OpenAIChatClient:
    kwargs = dict(
        model="gpt-4o-mini",
        api_key=API_KEY,
        base_url=api.base_url,
        sleep=sleeps.append,
        seed=7,
    )
    kwargs.update(overrides)
    return OpenAIChatClient(**kwargs)


CONVERSATION = [
    System("You resolve tickets."),
    User("Ticket T-001 ..."),
    Assistant(None, (ToolCallRequest("call_1", "lookup_customer", '{"customer_id": "CUST-1"}'),)),
    ToolResult("call_1", '{"plan": "pro"}'),
]


# --- what goes on the wire ---------------------------------------------------


@pytest.mark.asyncio
async def test_request_maps_every_message_type(api, sleeps) -> None:
    api.script.append((200, reply(calls=[("call_2", "lookup_customer", "{}")]), 0))
    await make(api, sleeps).complete(CONVERSATION, [LOOKUP])

    [request] = api.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == f"Bearer {API_KEY}"
    assert request["body"]["messages"] == [
        {"role": "system", "content": "You resolve tickets."},
        {"role": "user", "content": "Ticket T-001 ..."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_customer", "arguments": '{"customer_id": "CUST-1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"plan": "pro"}'},
    ]


@pytest.mark.asyncio
async def test_request_forces_exactly_one_tool_call_per_turn(api, sleeps) -> None:
    """ReAct one step at a time: a tool is required, and only one."""
    api.script.append((200, reply(calls=[("c", "lookup_customer", "{}")]), 0))
    await make(api, sleeps).complete(CONVERSATION, [LOOKUP])

    body = api.requests[0]["body"]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Fetch the account record.",
                "parameters": LOOKUP.parameters,
            },
        }
    ]
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False


@pytest.mark.asyncio
async def test_request_pins_model_temperature_and_seed(api, sleeps) -> None:
    api.script.append((200, reply(calls=[("c", "lookup_customer", "{}")]), 0))
    await make(api, sleeps, temperature=0.0, seed=7).complete(CONVERSATION, [LOOKUP])

    body = api.requests[0]["body"]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.0
    assert body["seed"] == 7


# --- what comes back -----------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_carries_calls_usage_and_the_served_model(api, sleeps) -> None:
    api.script.append(
        (200, reply(calls=[("call_9", "lookup_customer", '{"customer_id": "CUST-1042"}')],
                    prompt_tokens=200, completion_tokens=12), 0)
    )
    completion = await make(api, sleeps).complete(CONVERSATION, [LOOKUP])

    assert completion.tool_calls == (
        ToolCallRequest("call_9", "lookup_customer", '{"customer_id": "CUST-1042"}'),
    )
    assert completion.usage.input_tokens == 200
    assert completion.usage.output_tokens == 12
    assert completion.served_model == "gpt-4o-mini-2024-07-18"
    assert completion.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_arguments_are_passed_through_unparsed(api, sleeps) -> None:
    """Broken JSON is the loop's failure case to handle, not the adapter's."""
    api.script.append((200, reply(calls=[("c", "lookup_customer", '{"customer_id": ')]), 0))
    completion = await make(api, sleeps).complete(CONVERSATION, [LOOKUP])
    assert completion.tool_calls[0].arguments == '{"customer_id": '


@pytest.mark.asyncio
async def test_a_text_only_reply_has_no_tool_calls(api, sleeps) -> None:
    api.script.append((200, reply(text="I think...", finish_reason="stop"), 0))
    completion = await make(api, sleeps).complete(CONVERSATION, [LOOKUP])
    assert completion.tool_calls == ()
    assert completion.text == "I think..."


# --- failures ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limits_are_retried_with_backoff(api, sleeps) -> None:
    api.script += [
        (429, {"error": {"message": "slow down"}}, 0),
        (500, {"error": {"message": "boom"}}, 0),
        (200, reply(calls=[("c", "lookup_customer", "{}")]), 0),
    ]
    completion = await make(api, sleeps, backoff_base=0.5).complete(CONVERSATION, [LOOKUP])
    assert completion.tool_calls
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_retries_give_up_as_an_llm_error(api, sleeps) -> None:
    api.script += [(503, {"error": {"message": "down"}}, 0)] * 3
    with pytest.raises(LLMError, match="3 attempts"):
        await make(api, sleeps, max_attempts=3).complete(CONVERSATION, [LOOKUP])


@pytest.mark.asyncio
async def test_a_rejected_request_is_not_retried_and_hides_the_key(api, sleeps) -> None:
    api.script.append((401, {"error": {"message": f"Incorrect API key {API_KEY}"}}, 0))
    with pytest.raises(LLMError, match="401") as excinfo:
        await make(api, sleeps).complete(CONVERSATION, [LOOKUP])
    assert API_KEY not in str(excinfo.value)
    assert len(api.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"<html>bad gateway</html>", id="not-json"),
        pytest.param({"object": "chat.completion", "model": "m"}, id="no-choices"),
        pytest.param(
            {**reply(calls=[("c", "lookup_customer", "{}")]), "usage": None}, id="no-usage"
        ),
        pytest.param(
            json.loads(
                json.dumps(reply(calls=[("c", "lookup_customer", "{}")])).replace(
                    '"name": "lookup_customer"', '"name": 5'
                )
            ),
            id="non-string-name",
        ),
    ],
)
@pytest.mark.asyncio
async def test_an_unreadable_response_is_an_llm_error(api, sleeps, body) -> None:
    api.script.append((200, body, 0))
    with pytest.raises(LLMError, match="malformed"):
        await make(api, sleeps).complete(CONVERSATION, [LOOKUP])


def test_missing_api_key_fails_at_construction(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatClient(model="gpt-4o-mini")
