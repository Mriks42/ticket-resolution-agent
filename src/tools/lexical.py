"""Deterministic TF-IDF scoring over a small document set.

This is the retrieval backend for `retrieve_policy` and `search_past_tickets`
until pgvector lands. It is lexical, not semantic: it matches words, and a
query that paraphrases a clause without sharing vocabulary with it will miss.
That limitation is real and belongs in the eval numbers rather than being
papered over — a miss must surface as an empty result, so the retrieval
guardrail and the "retrieval returns nothing relevant" failure case both have
something honest to fire on.

Two properties matter downstream:

- **Deterministic.** Same corpus and query give the same ranking, ties broken
  by document id. Eval runs have to be reproducible.
- **Zero means zero.** A document sharing no term with the query scores 0 and
  is not returned. Never return a weak best guess to avoid an empty list.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")

# Small and deliberately conservative: these carry no signal in a 104-clause
# policy corpus, and dropping more would risk losing terms like "no" or "not"
# that flip a clause's meaning.
_STOPWORDS = frozenset(
    """
    a an the and or of to in on at by for with from is are was were be been
    this that these those it its as if then than there their they them we you
    i my our your do does did can could would should will shall may might
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords dropped, trivially de-pluralized."""
    tokens = []
    for raw in _WORD.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        # Crude singularization so "charges"/"charge" and "days"/"day" match.
        # Not a stemmer; it only has to work on policy prose.
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


@dataclass(frozen=True)
class Scored:
    doc_id: str
    score: float


class LexicalIndex:
    """TF-IDF index over `{doc_id: text}`. Built once, queried many times."""

    def __init__(self, documents: dict[str, str]) -> None:
        self._terms: dict[str, Counter[str]] = {
            doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()
        }
        n = len(self._terms) or 1
        doc_freq: Counter[str] = Counter()
        for counts in self._terms.values():
            doc_freq.update(counts.keys())
        # Smoothed IDF, always positive, so a term present in every document
        # still contributes a little rather than zeroing the score out.
        self._idf = {
            term: math.log(1.0 + n / (1 + df)) for term, df in doc_freq.items()
        }
        self._norm = {
            doc_id: math.sqrt(sum(c * c for c in counts.values())) or 1.0
            for doc_id, counts in self._terms.items()
        }

    def search(self, query: str, limit: int | None = None) -> list[Scored]:
        """Documents sharing at least one term with `query`, best first."""
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        hits: list[Scored] = []
        for doc_id, counts in self._terms.items():
            score = sum(
                counts[term] * self._idf.get(term, 0.0)
                for term in query_terms
                if term in counts
            )
            if score > 0:
                hits.append(Scored(doc_id, score / self._norm[doc_id]))

        # doc_id as the tie-break keeps the ranking stable across runs.
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits if limit is None else hits[:limit]
