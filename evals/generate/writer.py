"""The writer: one customer message per spec, from a model that is not the agent's.

Only the wording comes from the model. The spec has already fixed the facts,
the account and the label, and `checks.py` enforces the facts afterwards. The
writer is gpt-4o rather than the agent's gpt-4o-mini, so the agent is not
graded on prose its own model wrote.

The style examples are written for this file, not taken from the held-out
set, so the synthetic set does not echo held-out wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evals.generate.scenarios import REGISTERS, Spec
from src.agent.llm import System, User
from src.agent.openai_chat import OpenAIChatClient
from src.agent.pricing import SpendGuard

WRITER_MODEL = "gpt-4o"
#: Varied enough that 25 messages do not read alike. Each attempt gets its own
#: seed, which keeps a rerun close, but the committed batch is the record.
WRITER_TEMPERATURE = 0.9

SYSTEM = """\
You write customer emails for a test set that evaluates an AI support agent. The \
product is a subscription project-management tool with Free, Pro and Team plans.

Write only the body of one email, as the customer would type it: no subject line, \
no headers, no signature block beyond an optional first name.

Rules:
- Tell the situation in the customer's own words, and include every fact under MUST \
INCLUDE naturally.
- Never include anything under MUST NOT INCLUDE, not even indirectly.
- Customers do not know the policies. Never mention policy names, clause numbers, \
approval steps, or what support should decide.
- Do not add facts that would change the situation: no extra charges, disputes, legal \
threats, security problems or other people unless the situation says so.
- Between 30 and 160 words.
- Match the writing style exactly, including its typos and structure if it has them."""

STYLE_EXAMPLES = {
    "clean": (
        "Hello, we would like to understand how billing works when we add people partway "
        "through the month. Is the new seat charged immediately, or on our next invoice? "
        "Thank you in advance."
    ),
    "casual": (
        "quick one - if I add two seats today are we billed now or at renewal? trying to "
        "decide whether to wait a week. cheers"
    ),
    "messy": (
        "hey so we added 2 ppl today i think?? and now theres a charge on the card already "
        "which i wasnt expecting tbh, is that right or did something go wrong. renewal is "
        "the 3rd i thought"
    ),
    "buried": (
        "Hi there, hope your week is going better than ours - we've had a server migration "
        "and half the team is off with the flu. Anyway, while I was going through invoices "
        "for our finance close I noticed we were charged for two extra seats on Tuesday, "
        "which I think is from the contractors joining. Separately, is there a dark mode "
        "on the roadmap? People keep asking me."
    ),
}


@dataclass(frozen=True)
class Persona:
    """Who is writing, as the writer model is told it."""

    name: str
    role: str
    company: str


@dataclass(frozen=True)
class Draft:
    body: str
    served_model: str


class Writer(Protocol):
    async def write(self, spec: Spec, persona: Persona, seed: int) -> Draft: ...


def brief(spec: Spec, persona: Persona) -> str:
    """The user turn: everything the writer needs, and nothing about the label."""
    plan = f"{spec.plan.title()}, {spec.seats} seat{'s' if spec.seats != 1 else ''}"
    if spec.billing_cycle:
        plan += f", billed {spec.billing_cycle}"
    lines = [
        f"Writer: {persona.name}, {persona.role}, at {persona.company}.",
        f"Their plan: {plan}. Mention it only if it comes up naturally.",
        f"Situation: {spec.situation}",
        "MUST INCLUDE:",
        *(f"- {' or '.join(repr(o) for o in options)}" for options in spec.must_mention),
    ]
    if spec.must_not_mention:
        lines += ["MUST NOT INCLUDE:", *(f"- {phrase!r}" for phrase in spec.must_not_mention)]
    lines += [
        f"Writing style: {spec.register}. {REGISTERS[spec.register]}",
        f"An example of this style, about something else:\n{STYLE_EXAMPLES[spec.register]}",
    ]
    return "\n".join(lines)


class OpenAIWriter:
    def __init__(self, *, spend: SpendGuard) -> None:
        self._spend = spend

    async def write(self, spec: Spec, persona: Persona, seed: int) -> Draft:
        client = OpenAIChatClient(model=WRITER_MODEL, temperature=WRITER_TEMPERATURE, seed=seed)
        completion = await client.complete([System(SYSTEM), User(brief(spec, persona))], [])
        self._spend.charge(completion.usage)  # may raise SpendCapExceeded
        return Draft(body=(completion.text or "").strip(), served_model=completion.served_model)
