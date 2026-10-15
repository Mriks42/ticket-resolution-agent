"""The embedding client, exercised against a local HTTP server.

The server (`conftest.ScriptedAPI`) is real HTTP on a loopback port, scripted
per test. That exercises
the request the client actually puts on the wire and the parsing it actually
does, without a network dependency or an API key. Nothing here is in the eval
path: `evals/` always talks to the real endpoint.
"""

from __future__ import annotations

import pytest

from src.tools.embeddings import EmbeddingError, OpenAIEmbedder
from src.tools.errors import ToolError
from tests.conftest import ScriptedAPI

API_KEY = "sk-test-not-a-real-key"


def ok(vectors: list[list[float]], *, order: list[int] | None = None, tokens: int = 7):
    """A well-formed embeddings response, optionally with `data` reordered."""
    indices = order if order is not None else list(range(len(vectors)))
    return {
        "object": "list",
        "model": "test-model",
        "data": [
            {"object": "embedding", "index": i, "embedding": vectors[i]}
            for i in indices
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


@pytest.fixture
def sleeps() -> list[float]:
    return []


def make(api: ScriptedAPI, sleeps: list[float], **overrides) -> OpenAIEmbedder:
    kwargs = dict(
        model="test-model",
        dims=3,
        api_key=API_KEY,
        base_url=api.base_url,
        sleep=sleeps.append,
    )
    kwargs.update(overrides)
    return OpenAIEmbedder(**kwargs)


# --- the happy path ------------------------------------------------------


def test_embed_posts_model_and_inputs_with_bearer_auth(api, sleeps) -> None:
    api.script.append((200, ok([[1, 0, 0], [0, 1, 0]]), 0))
    make(api, sleeps).embed(["first", "second"])

    [request] = api.requests
    assert request["path"] == "/v1/embeddings"
    assert request["authorization"] == f"Bearer {API_KEY}"
    assert request["body"] == {"model": "test-model", "input": ["first", "second"]}


def test_vectors_come_back_in_input_order_whatever_order_the_api_uses(
    api, sleeps
) -> None:
    """The API tags each vector with `index`; position in `data` is not a
    contract. Trusting it would silently attach one clause's vector to
    another clause."""
    api.script.append((200, ok([[1, 0, 0], [0, 1, 0], [0, 0, 1]], order=[2, 0, 1]), 0))
    vectors = make(api, sleeps).embed(["a", "b", "c"])
    assert vectors == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_large_inputs_are_split_into_batches(api, sleeps) -> None:
    api.script += [
        (200, ok([[1, 0, 0], [0, 1, 0]]), 0),
        (200, ok([[0, 0, 1], [1, 1, 0]]), 0),
        (200, ok([[0, 1, 1]]), 0),
    ]
    vectors = make(api, sleeps, batch_size=2).embed(["a", "b", "c", "d", "e"])

    assert [r["body"]["input"] for r in api.requests] == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]


def test_empty_input_makes_no_request(api, sleeps) -> None:
    assert make(api, sleeps).embed([]) == []
    assert api.requests == []


def test_usage_is_accumulated_for_cost_reporting(api, sleeps) -> None:
    api.script += [(200, ok([[1, 0, 0]], tokens=5), 0), (200, ok([[0, 1, 0]], tokens=8), 0)]
    embedder = make(api, sleeps)
    embedder.embed(["a"])
    embedder.embed(["b"])
    assert embedder.tokens_used == 13
    assert embedder.requests_made == 2


# --- retries -------------------------------------------------------------


def test_rate_limits_and_server_errors_are_retried_with_backoff(api, sleeps) -> None:
    api.script += [
        (429, {"error": {"message": "slow down"}}, 0),
        (503, {"error": {"message": "unavailable"}}, 0),
        (200, ok([[1, 0, 0]]), 0),
    ]
    vectors = make(api, sleeps, backoff_base=0.5).embed(["a"])
    assert vectors == [[1, 0, 0]]
    assert sleeps == [0.5, 1.0]


def test_backoff_is_capped_and_gives_up_loudly(api, sleeps) -> None:
    api.script += [(500, {"error": {"message": "boom"}}, 0)] * 5
    embedder = make(api, sleeps, max_attempts=5, backoff_base=0.5, backoff_cap=1.0)

    with pytest.raises(EmbeddingError, match="5 attempts"):
        embedder.embed(["a"])
    assert sleeps == [0.5, 1.0, 1.0, 1.0]
    assert len(api.requests) == 5


def test_a_timeout_is_retried(api, sleeps) -> None:
    """CLAUDE.md lists "tool times out" as its own failure case."""
    api.script += [(200, ok([[9, 9, 9]]), 0.6), (200, ok([[1, 0, 0]]), 0)]
    vectors = make(api, sleeps, timeout=0.2).embed(["a"])
    assert vectors == [[1, 0, 0]]
    assert len(sleeps) == 1


def test_client_errors_are_not_retried(api, sleeps) -> None:
    """A bad key or a bad request will not get better by asking again."""
    api.script.append((401, {"error": {"message": "Incorrect API key provided"}}, 0))
    with pytest.raises(EmbeddingError, match="401"):
        make(api, sleeps).embed(["a"])
    assert len(api.requests) == 1
    assert sleeps == []


def test_errors_never_echo_the_api_key(api, sleeps) -> None:
    api.script.append((401, {"error": {"message": f"bad key {API_KEY}"}}, 0))
    with pytest.raises(EmbeddingError) as excinfo:
        make(api, sleeps).embed(["a"])
    assert API_KEY not in str(excinfo.value)


def test_embedding_errors_are_tool_errors(api, sleeps) -> None:
    """So the MCP server reports them as an `isError` result the agent can
    observe and plan around, rather than crashing the run."""
    assert issubclass(EmbeddingError, ToolError)


# --- malformed output ----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param({"object": "list"}, id="no-data"),
        pytest.param(ok([[1, 0, 0]]), id="too-few-vectors"),
        pytest.param(ok([[1, 0], [0, 1]]), id="wrong-dimensions"),
        pytest.param(ok([[1, 0, 0], [0, 1, 0]], order=[0, 0]), id="duplicate-index"),
        pytest.param(ok([[1, 0, "x"], [0, 1, 0]]), id="non-numeric"),
    ],
)
def test_malformed_output_raises_rather_than_being_indexed(api, sleeps, body) -> None:
    """A wrong-shaped vector stored once poisons every ranking after it."""
    api.script.append((200, body, 0))
    with pytest.raises(EmbeddingError, match="malformed"):
        make(api, sleeps).embed(["a", "b"])


# --- construction ----------------------------------------------------------


def test_missing_api_key_fails_at_construction(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(model="text-embedding-3-small")


def test_api_key_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", API_KEY)
    assert OpenAIEmbedder(model="text-embedding-3-small").dims == 1536


def test_unknown_model_needs_explicit_dims() -> None:
    with pytest.raises(ValueError, match="dims"):
        OpenAIEmbedder(model="some-future-model", api_key=API_KEY)
