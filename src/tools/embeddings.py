"""Text embeddings over HTTP, stdlib only.

The dense retrieval backend needs one thing from a model provider: a vector
per string. That is a single POST, so it is made with `urllib` rather than a
provider SDK, for the same reason ADR 0002 hand-rolls the MCP server — the
surface is small enough that a dependency would cost more than it saves.

`Embedder` is the provider-agnostic seam. `OpenAIEmbedder` is the one
implementation; the model name is a config value, and it is recorded with
every eval report because rankings under different models are not comparable
(ADR 0003).

Failure behaviour follows CLAUDE.md:

- **Retries** use exponential backoff with a cap, and only for failures that
  can plausibly clear on their own: rate limits, 5xx, timeouts, dropped
  connections. A 4xx is the request's fault and is raised at once.
- **Malformed output is an error, never a best effort.** A vector of the wrong
  length or attached to the wrong index, once stored, silently corrupts every
  ranking that touches it.
- Errors are `ToolError`s, so a failed embedding call reaches the agent as an
  observation rather than taking the run down.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from http.client import HTTPException
from typing import Any, Protocol

from src.tools.errors import ToolError

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

#: Output dimensions per model, so a wrong-length vector is caught on arrival.
KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

DEFAULT_BASE_URL = "https://api.openai.com/v1"

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class EmbeddingError(ToolError):
    """The provider could not, or did not properly, return embeddings."""


class Embedder(Protocol):
    model: str
    dims: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """`POST /embeddings` against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dims: int | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        batch_size: int = 256,
        max_attempts: int = 4,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if dims is None:
            if model not in KNOWN_DIMS:
                raise ValueError(
                    f"unknown embedding model {model!r}: pass dims explicitly "
                    f"or use one of {sorted(KNOWN_DIMS)}"
                )
            dims = KNOWN_DIMS[model]

        key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set; dense retrieval cannot embed a query "
                "without it. Add it to .env or the environment."
            )

        self.model = model
        self.dims = dims
        self._key = key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._timeout = timeout
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._sleep = sleep

        #: Cumulative, for cost reporting. Tokens are as the provider counted them.
        self.tokens_used = 0
        self.requests_made = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(list(texts[start : start + self._batch_size])))
        return vectors

    # --- one request -----------------------------------------------------

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        last_failure = ""
        for attempt in range(self._max_attempts):
            if attempt:
                self._sleep(min(self._backoff_cap, self._backoff_base * 2 ** (attempt - 1)))
            self.requests_made += 1
            request = urllib.request.Request(
                self._url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                last_failure = f"HTTP {exc.code}: {self._provider_message(exc)}"
                if exc.code not in _RETRYABLE_STATUS:
                    raise EmbeddingError(f"embeddings request rejected, {last_failure}") from None
                continue
            except (urllib.error.URLError, socket.timeout, ConnectionError, HTTPException) as exc:
                reason = getattr(exc, "reason", exc)
                last_failure = f"{type(reason).__name__}: {reason}"
                continue
            return self._parse(raw, len(batch))

        raise EmbeddingError(
            f"embeddings request failed after {self._max_attempts} attempts, "
            f"last error {last_failure}"
        )

    def _provider_message(self, exc: urllib.error.HTTPError) -> str:
        try:
            message = json.loads(exc.read())["error"]["message"]
        except Exception:
            message = exc.reason or "no detail"
        return str(message).replace(self._key, "[redacted]")[:300]

    def _parse(self, raw: bytes, expected: int) -> list[list[float]]:
        try:
            body: Any = json.loads(raw)
            data = body["data"]
        except (ValueError, KeyError, TypeError):
            raise EmbeddingError("malformed embeddings response: no `data` list") from None

        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else type(data).__name__
            raise EmbeddingError(
                f"malformed embeddings response: asked for {expected} vectors, got {got}"
            )

        by_index: dict[int, list[float]] = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(index, int) or index in by_index or not 0 <= index < expected:
                raise EmbeddingError(
                    f"malformed embeddings response: bad or duplicate index {index!r}"
                )
            if not isinstance(vector, list) or len(vector) != self.dims:
                got = len(vector) if isinstance(vector, list) else type(vector).__name__
                raise EmbeddingError(
                    f"malformed embeddings response: vector {index} has "
                    f"{got} dimensions, expected {self.dims}"
                )
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector):
                raise EmbeddingError(
                    f"malformed embeddings response: vector {index} is not all numbers"
                )
            by_index[index] = [float(x) for x in vector]

        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            self.tokens_used += usage["total_tokens"]

        return [by_index[i] for i in range(expected)]
