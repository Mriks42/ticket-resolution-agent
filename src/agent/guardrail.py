"""The retrieval guardrail: every policy fact in a reply must be grounded.

CLAUDE.md: "Any policy claim in a drafted reply must be checkable against the
passages actually retrieved in that run. A claim with no supporting retrieved
passage is a hallucinated-policy event." This module is that check, and it is
deterministic by decision (ADR 0007): no model judges another model's claims,
so the hallucinated-policy rate does not rest on model judgement.

**What it checks.** A reply is split into sentences, and each sentence's
checkable facts are extracted and normalised:

- money        USD 50, $1,584, 15 dollars
- durations    14 days, 30-day, two weeks (= 14 days), 5–10 business days,
               4 business hours, 12 months
- counts       3 seats, 3 projects
- storage      500 MB, 50 GB
- percentages  20%

plus every clause id the sentence cites.

**What counts as grounded.** A fact is grounded if it appears, with the same
kind, value and unit, in a passage retrieved in this run, or in the customer's
own data: the ticket body, the account record, past tickets, and the amounts
of refunds proposed in the run. A sentence that cites a retrieved clause is
held to *that* clause. "The specific passage that supports it" is what
clause-level IDs exist for.

**Violations**, each a hallucinated-policy event:

- `citation_not_retrieved`  the sentence cites a clause no retrieval returned
- `unsupported_fact`        a fact grounded nowhere
- `miscited_fact`           a fact from real, retrieved policy, attributed to
                            a cited clause that does not contain it

**What it cannot see.** A claim with no number in it — "annual plans can be
refunded at any time" — has nothing to check, and passes. So does a correct
figure the agent computed, such as a proration, which appears in no source;
that one fails the other way, flagged though true. Both are measured on the
reviewed reply set (`python -m evals.guardrail_eval`), not assumed away.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "ninety": 90,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(_WORDS) + r")"
_AMOUNT = r"(\d[\d,]*(?:\.\d+)?)"
_UNIT = r"(day|hour|week|month|year)s?\b"
_ADJ = r"(?:(business|working|calendar|consecutive|whole)\s+)?"

_MONEY = [
    re.compile(r"\bUSD\s*" + _AMOUNT, re.IGNORECASE),
    re.compile(r"\$\s*" + _AMOUNT),
    re.compile(_AMOUNT + r"\s*(?:USD|dollars?)\b", re.IGNORECASE),
]
_RANGE = re.compile(r"\b(\d+)\s*[–-]\s*(\d+)[\s-]+" + _ADJ + _UNIT, re.IGNORECASE)
_DURATION = re.compile(r"(?<![\w.-])" + _NUM + r"[\s-]+" + _ADJ + _UNIT, re.IGNORECASE)
_COUNT = re.compile(
    r"(?<![\w.-])" + _NUM + r"[\s-]+(?:(?:billable|additional|extra|more)\s+)?(seat|project)s?\b",
    re.IGNORECASE,
)
_STORAGE = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)\s*(MB|GB|TB)\b", re.IGNORECASE)
_PERCENT = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)

_CLAUSE_ID = re.compile(r"\b[A-Z]{3}-\d{3}\.\d+\b")
#: A sentence ends at . ! or ? followed by whitespace — so "50.01" and
#: "REF-001.3" never split, since neither is followed by a space mid-token.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Fact:
    kind: str
    value: Decimal
    unit: str
    #: As written, for the violation message. Not part of equality: "two
    #: weeks" and "14 days" are the same fact.
    text: str = field(default="", compare=False)


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str
    sentence: str


@dataclass(frozen=True)
class Sources:
    #: clause_id -> text, for every passage the model was shown in this run.
    passages: Mapping[str, str]
    #: The customer's own words and records: ticket body, account, history.
    context: tuple[str, ...] = ()
    #: Facts with no text to extract them from, e.g. refund amounts proposed.
    context_facts: frozenset[Fact] = frozenset()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def cited_clause_ids(text: str) -> set[str]:
    return set(_CLAUSE_ID.findall(text))


def extract_facts(text: str) -> set[Fact]:
    found: set[Fact] = set()
    for pattern in _MONEY:
        for m in pattern.finditer(text):
            found.add(Fact("money", _money(m.group(1)), "usd", m.group(0).strip()))
    for m in _RANGE.finditer(text):
        unit = _duration_unit(m.group(3), m.group(4))
        for number in (m.group(1), m.group(2)):
            value, norm = _scale(Decimal(number), unit)
            found.add(Fact("duration", value, norm, m.group(0).strip()))
    for m in _DURATION.finditer(text):
        value, unit = _scale(_number(m.group(1)), _duration_unit(m.group(2), m.group(3)))
        found.add(Fact("duration", value, unit, m.group(0).strip()))
    for m in _COUNT.finditer(text):
        found.add(Fact("count", _number(m.group(1)), m.group(2).lower(), m.group(0).strip()))
    for m in _STORAGE.finditer(text):
        found.add(Fact("storage", Decimal(m.group(1)), m.group(2).lower(), m.group(0).strip()))
    for m in _PERCENT.finditer(text):
        found.add(Fact("percent", Decimal(m.group(1)), "percent", m.group(0).strip()))
    return found


def account_grounding(account: Mapping | None) -> tuple[list[str], set[Fact]]:
    """What a `lookup_customer` record grounds: its dollar figures and seat
    count as facts, its past tickets as text. Shared by the loop and
    `evals/guardrail_eval.py`, so both ground a reply the same way."""
    if not isinstance(account, Mapping):
        return [], set()
    found: set[Fact] = set()
    for key, value in account.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key.endswith("_usd"):
            found.add(usd(value))
        elif key == "seats":
            found.add(Fact("count", Decimal(int(value)), "seat"))
    return history_text(account.get("past_tickets")), found


def history_text(items) -> list[str]:
    """Summary and resolution text of past tickets, from either a customer
    record or a `search_past_tickets` result."""
    if not isinstance(items, list):
        return []
    return [str(i.get(k, "")) for i in items if isinstance(i, dict) for k in ("summary", "resolution")]


def usd(amount) -> Fact:
    return Fact("money", Decimal(str(amount)).quantize(Decimal("0.01")), "usd")


def facts_of(values: Iterable[str]) -> set[Fact]:
    found: set[Fact] = set()
    for value in values:
        found |= extract_facts(value)
    return found


def check(text: str, sources: Sources) -> list[Violation]:
    by_clause = {cid: extract_facts(body) for cid, body in sources.passages.items()}
    all_policy = set().union(*by_clause.values()) if by_clause else set()
    customer = facts_of(sources.context) | set(sources.context_facts)

    violations: list[Violation] = []
    for sentence in split_sentences(text):
        cited = cited_clause_ids(sentence)
        for clause_id in sorted(cited - sources.passages.keys()):
            violations.append(Violation(
                "citation_not_retrieved",
                f"cites {clause_id}, which no retrieval in this run returned",
                sentence,
            ))
        backing = sorted(cited & sources.passages.keys())
        for fact in sorted(extract_facts(sentence), key=lambda f: (f.kind, f.value, f.unit)):
            if fact in customer:
                continue
            if backing:
                if any(fact in by_clause[cid] for cid in backing):
                    continue
                if fact in all_policy:
                    elsewhere = sorted(cid for cid, fs in by_clause.items() if fact in fs)
                    violations.append(Violation(
                        "miscited_fact",
                        f"'{fact.text}' is not in {', '.join(backing)}; "
                        f"it is in {', '.join(elsewhere)}",
                        sentence,
                    ))
                    continue
            elif fact in all_policy:
                continue
            violations.append(Violation(
                "unsupported_fact",
                f"'{fact.text}' is in no passage retrieved in this run and not in the customer's data",
                sentence,
            ))
    return violations


def _number(raw: str) -> Decimal:
    return Decimal(_WORDS[raw.lower()]) if raw.lower() in _WORDS else Decimal(raw)


def _money(raw: str) -> Decimal:
    return Decimal(raw.replace(",", "")).quantize(Decimal("0.01"))


def _duration_unit(adjective: str | None, unit: str) -> str:
    unit = unit.lower()
    if adjective and adjective.lower() in ("business", "working") and unit in ("day", "hour"):
        return f"business_{unit}"
    return unit


def _scale(value: Decimal, unit: str) -> tuple[Decimal, str]:
    """Weeks are days. Nothing else is folded: a month is not 30 days."""
    if unit == "week":
        return value * 7, "day"
    return value, unit
