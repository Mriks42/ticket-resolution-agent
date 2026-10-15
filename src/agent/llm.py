"""The provider-agnostic seam between the agent loop and a model.

The loop speaks only these types. An adapter (`src/agent/openai_chat.py`)
translates them to one provider's wire format, so swapping providers means
writing another adapter and never touching the loop.

Tool-call arguments stay a raw string here on purpose. Parsing them is the
loop's job, because "the LLM returns unparseable tool-call JSON" is one of
CLAUDE.md's failure cases and has to surface as exactly that, not be repaired
or swallowed inside an adapter.

Usage is mandatory on every completion. The token budget and the cost metric
both read it, and a completion that does not say what it cost would let a run
spend without being counted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, Union


@dataclass(frozen=True)
class FunctionSchema:
    """A tool as the model sees it: a name, prose, and a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the model asked for. `arguments` is unparsed JSON."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class System:
    content: str


@dataclass(frozen=True)
class User:
    content: str


@dataclass(frozen=True)
class Assistant:
    text: str | None
    tool_calls: tuple[ToolCallRequest, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolResult:
    """What the loop observed for one `ToolCallRequest`, sent back verbatim."""

    call_id: str
    content: str


Message = Union[System, User, Assistant, ToolResult]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Completion:
    text: str | None
    tool_calls: tuple[ToolCallRequest, ...]
    usage: Usage
    #: The model that actually answered, as the provider reports it. Often a
    #: dated snapshot of the requested alias, which is what an eval record
    #: needs in order to be comparable later.
    served_model: str
    finish_reason: str | None = None


class LLMError(Exception):
    """The provider did not produce a usable completion: rejected, retries
    exhausted, or a response the adapter could not read. The loop cannot plan
    without one, so this ends the run in FAILED."""


class LLMClient(Protocol):
    model: str

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[FunctionSchema]
    ) -> Completion: ...
