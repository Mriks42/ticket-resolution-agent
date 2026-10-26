"""What a completion cost, and the hard cap on what an eval run may spend.

Prices are per million tokens, from OpenAI's model pages, checked 2026-10-14:
https://developers.openai.com/api/docs/models/gpt-4o-mini

All input is priced at the full input rate. OpenAI discounts cached input, so
this overstates cost slightly when the prompt prefix is cached — the safe
direction for a cap, and a known overstatement in the cost metric until usage
reports cached tokens separately.

A model missing from `PRICES` has no known cost, so the spend guard refuses to
build for it rather than running uncapped. The same holds for embedding
models and `EMBEDDING_PRICES`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from src.agent.llm import Usage


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


PRICES: dict[str, Price] = {
    "gpt-4o-mini": Price(input_per_mtok=0.15, output_per_mtok=0.60),
}

#: USD per million tokens, from OpenAI's model page, checked 2026-10-26:
#: https://developers.openai.com/api/docs/models/text-embedding-3-small
EMBEDDING_PRICES: dict[str, float] = {
    "text-embedding-3-small": 0.02,
}


class SpendCapExceeded(RuntimeError):
    """Raised out of the run, deliberately not caught by the loop. An eval
    that hits its cap stops; it does not report partial numbers."""


class SpendGuard:
    """Cumulative across every run on one agent — the cap is per eval run,
    not per ticket."""

    def __init__(self, *, cap_usd: float, price: Price) -> None:
        self.cap_usd = cap_usd
        self.price = price
        self.spent_usd = 0.0

    @classmethod
    def for_model(cls, model: str, cap_usd: float) -> SpendGuard:
        if model not in PRICES:
            raise ValueError(
                f"no price on record for {model!r}; add it to src/agent/pricing.py "
                "before running it, so the spend cap can be enforced"
            )
        return cls(cap_usd=cap_usd, price=PRICES[model])

    def charge(self, usage: Usage) -> float:
        cost = self.price.cost(usage)
        self.add(cost)
        return cost

    def add(self, cost_usd: float) -> None:
        """Spend already priced elsewhere, such as embeddings."""
        self.spent_usd += cost_usd
        if self.spent_usd > self.cap_usd:
            raise SpendCapExceeded(
                f"spent ${self.spent_usd:.4f}, over the ${self.cap_usd:.2f} cap for this run"
            )


class Metered(Protocol):
    model: str
    #: Cumulative, as the provider counted them.
    tokens_used: int


class EmbeddingMeter:
    """What the embedders behind the tools have spent, at list price.

    The loop reads it before and after a run and charges the difference to
    that run, so embedding done while building an index, before any ticket,
    is charged to none. Exact only while runs are sequential, as an eval's
    are (ADR 0009).
    """

    def __init__(self, embedders: Sequence[Metered]) -> None:
        missing = sorted({e.model for e in embedders} - set(EMBEDDING_PRICES))
        if missing:
            raise ValueError(
                f"no embedding price on record for {missing}; add it to "
                "src/agent/pricing.py so a run's cost includes its embeddings"
            )
        self._embedders = list(embedders)

    def spent_usd(self) -> float:
        return sum(e.tokens_used * EMBEDDING_PRICES[e.model] for e in self._embedders) / 1_000_000
