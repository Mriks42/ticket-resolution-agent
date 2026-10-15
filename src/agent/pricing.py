"""What a completion cost, and the hard cap on what an eval run may spend.

Prices are per million tokens, from OpenAI's model pages, checked 2026-10-14:
https://developers.openai.com/api/docs/models/gpt-4o-mini

All input is priced at the full input rate. OpenAI discounts cached input, so
this overstates cost slightly when the prompt prefix is cached — the safe
direction for a cap, and a known overstatement in the cost metric until usage
reports cached tokens separately.

A model missing from `PRICES` has no known cost, so the spend guard refuses to
build for it rather than running uncapped.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        self.spent_usd += cost
        if self.spent_usd > self.cap_usd:
            raise SpendCapExceeded(
                f"spent ${self.spent_usd:.4f}, over the ${self.cap_usd:.2f} cap for this run"
            )
        return cost
