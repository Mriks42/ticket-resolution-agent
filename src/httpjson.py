"""POST JSON with capped exponential backoff, stdlib only.

Shared by the embedding client (`src/tools/embeddings.py`) and the LLM client
(`src/agent/openai_chat.py`). Both make one kind of request to a model
provider, and both need the same failure behaviour from CLAUDE.md:

- **Retry** only what can plausibly clear on its own: rate limits, 5xx,
  timeouts, dropped connections. Backoff doubles from `backoff_base` and is
  capped at `backoff_cap`.
- **Do not retry** a 4xx. A bad key or a bad request will not improve by
  asking again, and retrying it just spends the budget.
- **Never echo a secret.** Provider error messages are scrubbed of every
  string in `redact` before they reach an exception.

Callers turn the two exceptions here into their own error types; this module
knows nothing about embeddings or chat.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http.client import HTTPException

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class HTTPRejected(Exception):
    """A non-retryable HTTP error. The request itself was the problem."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class HTTPRetriesExhausted(Exception):
    """Every attempt failed with something retryable."""

    def __init__(self, attempts: int, last_failure: str) -> None:
        super().__init__(f"failed after {attempts} attempts, last error {last_failure}")
        self.attempts = attempts
        self.last_failure = last_failure


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    backoff_base: float = 0.5
    backoff_cap: float = 8.0

    def delay(self, attempt: int) -> float:
        """Sleep before `attempt` (1-based retries; attempt 0 never sleeps)."""
        return min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))


def post_json(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str],
    timeout: float,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    redact: Iterable[str] = (),
    on_attempt: Callable[[], None] | None = None,
) -> bytes:
    """The raw response body of the first successful attempt."""
    body = json.dumps(payload).encode("utf-8")
    secrets = [s for s in redact if s]
    last_failure = ""
    for attempt in range(policy.max_attempts):
        if attempt:
            sleep(policy.delay(attempt))
        if on_attempt is not None:
            on_attempt()
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            message = _scrub(_provider_message(exc), secrets)
            if exc.code not in RETRYABLE_STATUS:
                raise HTTPRejected(exc.code, message) from None
            last_failure = f"HTTP {exc.code}: {message}"
        except (urllib.error.URLError, socket.timeout, ConnectionError, HTTPException) as exc:
            reason = getattr(exc, "reason", exc)
            last_failure = _scrub(f"{type(reason).__name__}: {reason}", secrets)
    raise HTTPRetriesExhausted(policy.max_attempts, last_failure)


def _provider_message(exc: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(exc.read())["error"]["message"])[:300]
    except Exception:
        return str(exc.reason or "no detail")[:300]


def _scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text
