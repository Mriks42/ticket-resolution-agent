"""Agent loop configuration: model settings and the three budgets.

CLAUDE.md fixes the step budget's default at 8 and requires all three budgets
to be enforced in code and configurable. It does not fix the token or
wall-clock defaults; the values here are starting points, chosen so that a
normal 8-step run on gpt-4o-mini fits comfortably inside them, and they belong
in the step-budget sweep (4 / 8 / 16) once that ablation runs.

Exceeding any budget routes to ESCALATE, never to a silent failure.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from src.agent.openai_chat import DEFAULT_CHAT_MODEL, DEFAULT_SEED


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_CHAT_MODEL
    temperature: float = 0.0
    seed: int | None = DEFAULT_SEED

    #: Tool decisions per ticket — executed reads plus held gated calls.
    max_steps: int = 8
    #: Input plus output tokens across every LLM call in one ticket.
    max_tokens: int = 50_000
    max_seconds: float = 120.0

    #: Per tool call, before the executor's own retry.
    tool_timeout: float = 20.0
    #: Unparseable arguments, unknown tools or turns with no tool call. The
    #: model is told what was wrong the first time; this many ends the run.
    max_invalid_calls: int = 2
    #: An identical call (same tool, same arguments) is not re-executed; the
    #: model is told it already has the answer. This many repeats ends the run.
    max_repeats: int = 2

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tokens", "max_invalid_calls", "max_repeats"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("max_seconds", "tool_timeout"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AgentConfig:
        env = os.environ if env is None else env
        defaults = cls()

        def get(name: str, cast, default):
            raw = (env.get(name) or "").strip()
            if not raw:
                return default
            try:
                return cast(raw)
            except ValueError:
                raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from None

        return cls(
            model=get("LLM_MODEL", str, defaults.model),
            temperature=get("LLM_TEMPERATURE", float, defaults.temperature),
            seed=get("LLM_SEED", int, defaults.seed),
            max_steps=get("AGENT_MAX_STEPS", int, defaults.max_steps),
            max_tokens=get("AGENT_MAX_TOKENS", int, defaults.max_tokens),
            max_seconds=get("AGENT_MAX_WALL_CLOCK_SECONDS", float, defaults.max_seconds),
            tool_timeout=get("AGENT_TOOL_TIMEOUT_SECONDS", float, defaults.tool_timeout),
        )

    def budgets(self) -> dict:
        return {
            k: v
            for k, v in asdict(self).items()
            if k.startswith("max_") or k == "tool_timeout"
        }
