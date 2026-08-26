"""Measure `retrieve_policy` against the labels, with no agent loop involved.

    python -m evals.retrieval_baseline
    python -m evals.retrieval_baseline --out evals/report/retrieval.json

Every number in `docs/adr/0003-retrieval-approach.md` comes from here.

Retrieval quality is separable from agent quality, and measuring it on its own
is cheap: no LLM, no cost, no budgets. That makes this the regression check for
any change to the retrieval backend, and the baseline a new backend has to
beat.

**Document-level rollup.** Ticket labels carry document IDs
(`relevant_policy_ids`, e.g. `POL-REF-001`) while retrieval returns clause IDs
(`REF-001.3`). Retrieved clauses are therefore rolled up to their parent
document before comparison. This flatters the retriever slightly — returning
any clause of POL-REF-001 counts as retrieving it, even the wrong clause. Once
labels carry clause-level ground truth, tighten this.

**Recall, not precision.** What matters here is whether the governing policy
reached the agent at all. A passage the agent did not need is cheap; a clause
it never saw cannot be cited, and the guardrail will correctly refuse to let
it claim something unsupported.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evals.loader import POLICY_DIR, TICKETS_PATH, load_tickets
from evals.schema import Outcome, Ticket
from src.tools.policy import PolicyIndex, load_policy_index

#: Swept when choosing the default `limit` for `retrieve_policy`.
DEFAULT_K_SWEEP = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)

#: Rough characters-per-token. Only used to give the context cost a familiar
#: unit; it is an approximation and is labeled as one wherever it is printed.
CHARS_PER_TOKEN = 4


@dataclass
class KResult:
    k: int
    mean_recall: float
    full_recall_tickets: int
    esc_hits: int
    esc_total: int
    mean_chars: float
    per_ticket: dict[str, float] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)

    @property
    def mean_tokens(self) -> int:
        return round(self.mean_chars / CHARS_PER_TOKEN)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "mean_recall": round(self.mean_recall, 4),
            "full_recall_tickets": self.full_recall_tickets,
            "esc_001_retrieved": f"{self.esc_hits}/{self.esc_total}",
            "mean_chars_returned": round(self.mean_chars, 1),
            "approx_mean_tokens": self.mean_tokens,
            "per_ticket_recall": {t: round(r, 4) for t, r in self.per_ticket.items()},
            "missing_documents": self.missing,
        }


def measure_k(index: PolicyIndex, tickets: list[Ticket], k: int) -> KResult:
    recalls: dict[str, float] = {}
    missing: dict[str, list[str]] = {}
    chars: list[int] = []
    esc_hits = 0
    esc_total = 0

    for ticket in tickets:
        hits = index.search(ticket.body, k)
        retrieved_docs = {clause.policy_id for clause, _ in hits}
        chars.append(sum(len(clause.text) for clause, _ in hits))

        wanted = set(ticket.labels.relevant_policy_ids)
        recalls[ticket.ticket_id] = len(wanted & retrieved_docs) / len(wanted)
        if wanted - retrieved_docs:
            missing[ticket.ticket_id] = sorted(wanted - retrieved_docs)

        if "POL-ESC-001" in wanted:
            esc_total += 1
            esc_hits += "POL-ESC-001" in retrieved_docs

    return KResult(
        k=k,
        mean_recall=sum(recalls.values()) / len(recalls),
        full_recall_tickets=sum(1 for r in recalls.values() if r == 1.0),
        esc_hits=esc_hits,
        esc_total=esc_total,
        mean_chars=sum(chars) / len(chars),
        per_ticket=recalls,
        missing=missing,
    )


def clause_rank(index: PolicyIndex, query: str, clause_id: str) -> int | None:
    """1-based rank of a clause for a query, or None if it never surfaces."""
    ranked = [c.clause_id for c, _ in index.search(query, len(index))]
    return ranked.index(clause_id) + 1 if clause_id in ranked else None


def document_rank(
    index: PolicyIndex, query: str, policy_id: str
) -> tuple[int | None, int]:
    """1-based rank of a document's best-scoring clause, and how many clauses
    scored above zero at all.

    The second number is what makes "ranked last" checkable: only clauses
    sharing a term with the query are returned, so a rank equal to it means
    the document was the worst match that matched at all.
    """
    ranked = index.search(query, len(index))
    for position, (clause, _) in enumerate(ranked, 1):
        if clause.policy_id == policy_id:
            return position, len(ranked)
    return None, len(ranked)


def format_report(
    index: PolicyIndex, tickets: list[Ticket], results: list[KResult], default_k: int
) -> str:
    lines = [
        "",
        "retrieval baseline - lexical TF-IDF, ticket body as query",
        f"corpus     {len(index)} clauses",
        f"tickets    {len(tickets)}",
        "recall is over relevant_policy_ids, clauses rolled up to documents",
        "",
        f"{'k':>4}{'recall':>9}{'full':>8}{'ESC-001':>10}{'chars':>9}{'~tokens':>9}",
        "-" * 49,
    ]
    for r in results:
        marker = "  <- default" if r.k == default_k else ""
        lines.append(
            f"{r.k:>4}{r.mean_recall:>9.3f}{r.full_recall_tickets:>4}/{len(tickets):<3}"
            f"{r.esc_hits:>7}/{r.esc_total:<2}{r.mean_chars:>9.0f}{r.mean_tokens:>9}"
            f"{marker}"
        )

    lines += [
        "",
        "  recall   mean fraction of a ticket's relevant documents retrieved",
        "  full     tickets where every relevant document was retrieved",
        "  ESC-001  of the tickets POL-ESC-001 is relevant to, how many retrieved it",
        "  chars    mean characters of passage text returned per ticket",
        "  ~tokens  chars/4, an approximation for context-budget purposes only",
        "",
    ]

    at_default = next(r for r in results if r.k == default_k)
    worst = sorted(at_default.per_ticket.items(), key=lambda kv: kv[1])[:8]
    lines += [f"worst tickets at k={default_k}", ""]
    for ticket_id, recall in worst:
        gap = ", ".join(at_default.missing.get(ticket_id, [])) or "-"
        lines.append(f"  {ticket_id}  recall={recall:.2f}  missing: {gap}")

    lines += ["", "POL-ESC-001 rank on the pending_approval tickets", ""]
    for ticket in tickets:
        if ticket.labels.outcome is Outcome.PENDING_APPROVAL:
            rank, scoring = document_rank(index, ticket.body, "POL-ESC-001")
            if rank is None:
                lines.append(f"  {ticket.ticket_id}  never retrieved")
                continue
            last = "  (last scoring clause)" if rank == scoring else ""
            lines.append(
                f"  {ticket.ticket_id}  rank {rank} of {scoring} clauses "
                f"scoring above zero{last}"
            )

    lines += ["", "T-016, the planted REF-001.3 / CAN-002.2 conflict", ""]
    t016 = next((t for t in tickets if t.ticket_id == "T-016"), None)
    if t016 is not None:
        for clause_id in ("REF-001.3", "CAN-002.2"):
            rank = clause_rank(index, t016.body, clause_id)
            lines.append(f"  {clause_id}  rank {rank}")
        lines.append(
            "  both sides are needed for the agent to notice the contradiction"
        )

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=TICKETS_PATH)
    parser.add_argument("--policies", type=Path, default=POLICY_DIR)
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(DEFAULT_K_SWEEP),
        help="k values to sweep.",
    )
    parser.add_argument(
        "--default-k", type=int, default=None,
        help="Mark this k as the configured default. Defaults to the live value.",
    )
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)

    if args.default_k is None:
        from src.tools.catalog import DEFAULT_RETRIEVAL_LIMIT

        args.default_k = DEFAULT_RETRIEVAL_LIMIT

    tickets = load_tickets(args.dataset, args.policies)
    index = load_policy_index(args.policies)

    ks = sorted(set(args.k) | {args.default_k})
    results = [measure_k(index, tickets, k) for k in ks]

    print(format_report(index, tickets, results, args.default_k))

    if args.out:
        payload = {
            "backend": "lexical-tfidf",
            "clauses": len(index),
            "tickets": len(tickets),
            "default_k": args.default_k,
            "rollup": "clause ids rolled up to policy_id; labels are document-level",
            "sweep": [r.as_dict() for r in results],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report written to {args.out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
