"""Measure the guardrail against labelled reply drafts. No model, no cost.

    python -m evals.guardrail_eval
    python -m evals.guardrail_eval --out evals/report/guardrail.json

The hallucinated-policy rate `evals/run.py` reports is what the guardrail
found. This says how much to trust that: run the same check, with the same
grounding the loop uses, over `evals/dataset/guardrail_cases.json`, and compare
with labels that say what each reply truly contains.

**Case level.** A case is positive if its label lists any error; the
guardrail predicts positive if it reports any violation. Precision and recall
are over cases.

**Two recalls.** `blind_spot` labels mark real errors with no figure in them,
which a deterministic figure check cannot see by design (ADR 0007). Recall is
reported over every erroneous case, and again over the checkable ones only.
The first is what the metric can see in practice; the second says whether the
check does what it claims.

**Read the checkable numbers as an upper bound.** The cases were drafted
after the checker, by the same author. A checkable case is one the author
knew how to make checkable. Independent cases — written by the label owner, or
drawn from the agent's own replies — are what would turn this into a
measurement. Every disagreement is listed with its case, so the numbers can be
audited one row at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from evals.loader import CUSTOMERS_PATH, DATASET_DIR, TICKETS_PATH, load_customers, load_tickets
from src.agent.guardrail import Sources, account_grounding, check, usd
from src.tools.policy import load_clauses

CASES_PATH = DATASET_DIR / "guardrail_cases.json"
KINDS = frozenset({"unsupported_fact", "miscited_fact", "citation_not_retrieved", "blind_spot"})


@dataclass
class Outcome:
    case_id: str
    labelled: set[str]
    found: set[str]
    details: list[str]
    why: str

    @property
    def truly_wrong(self) -> bool:
        return bool(self.labelled)

    @property
    def flagged(self) -> bool:
        return bool(self.found)

    @property
    def checkable(self) -> bool:
        return bool(self.labelled - {"blind_spot"})


def load_cases(path: Path = CASES_PATH) -> tuple[list[dict], bool]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["cases"]
    clause_ids = {c.clause_id for c in load_clauses()}
    ticket_ids = {t.ticket_id for t in load_tickets(TICKETS_PATH)}
    seen: set[str] = set()
    for case in cases:
        cid = case["case_id"]
        if cid in seen:
            raise ValueError(f"{path.name}: duplicate case id {cid}")
        seen.add(cid)
        if case["ticket_id"] not in ticket_ids:
            raise ValueError(f"{cid}: unknown ticket {case['ticket_id']}")
        unknown = set(case["retrieved"]) - clause_ids
        if unknown:
            raise ValueError(f"{cid}: unknown clause ids {sorted(unknown)}")
        bad_kinds = {item["kind"] for item in case["label"]} - KINDS
        if bad_kinds:
            raise ValueError(f"{cid}: unknown label kinds {sorted(bad_kinds)}")
    return cases, bool(raw.get("reviewed", False))


def run_case(case: dict, tickets: dict, customers: dict, clause_text: dict) -> Outcome:
    """Ground the reply exactly as the loop would have, then check it."""
    ticket = tickets[case["ticket_id"]]
    context = [ticket.body]
    facts = {usd(amount) for amount in case["refunds"]}
    if case["account"]:
        texts, account_facts = account_grounding(customers[ticket.customer_id].model_dump(mode="json"))
        context += texts
        facts |= account_facts
    sources = Sources(
        passages={cid: clause_text[cid] for cid in case["retrieved"]},
        context=tuple(context),
        context_facts=frozenset(facts),
    )
    violations = check(case["reply"], sources)
    return Outcome(
        case_id=case["case_id"],
        labelled={item["kind"] for item in case["label"]},
        found={v.kind for v in violations},
        details=[f"{v.kind}: {v.detail}" for v in violations],
        why=case["why"],
    )


def summarise(outcomes: list[Outcome]) -> dict:
    tp = [o for o in outcomes if o.truly_wrong and o.flagged]
    fp = [o for o in outcomes if not o.truly_wrong and o.flagged]
    fn = [o for o in outcomes if o.truly_wrong and not o.flagged]
    tn = [o for o in outcomes if not o.truly_wrong and not o.flagged]
    checkable = [o for o in outcomes if o.checkable]
    caught = [o for o in checkable if o.flagged]
    kind_match = [o for o in tp if o.checkable and (o.labelled - {"blind_spot"}) <= o.found]

    def ratio(a, b):
        return round(len(a) / len(b), 4) if b else None

    return {
        "cases": len(outcomes),
        "true_positive": len(tp),
        "false_positive": len(fp),
        "false_negative": len(fn),
        "true_negative": len(tn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "recall_checkable": ratio(caught, checkable),
        "every_labelled_kind_found": ratio(kind_match, [o for o in tp if o.checkable]),
        "false_positives": [o.case_id for o in fp],
        "false_negatives": [o.case_id for o in fn],
    }


def format_report(outcomes: list[Outcome], summary: dict, reviewed: bool) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "",
        "guardrail - deterministic fact check, against labelled reply drafts",
        *([] if reviewed else ["** case labels UNREVIEWED: every number below is provisional **"]),
        "",
        f"  cases            {summary['cases']}",
        f"  flagged & wrong  {summary['true_positive']}    flagged & clean  {summary['false_positive']}",
        f"  missed & wrong   {summary['false_negative']}    passed & clean   {summary['true_negative']}",
        "",
        f"  precision                    {pct(summary['precision'])}",
        f"  recall, all errors           {pct(summary['recall'])}",
        f"  recall, checkable errors     {pct(summary['recall_checkable'])}   (upper bound: same author as the checker)",
        f"  right kind on caught cases   {pct(summary['every_labelled_kind_found'])}",
        "",
    ]
    by_id = {o.case_id: o for o in outcomes}
    if summary["false_positives"]:
        lines += ["flagged, but labelled clean", ""]
        for cid in summary["false_positives"]:
            o = by_id[cid]
            lines += [f"  {cid}  {o.why}", *(f"        {d}" for d in o.details)]
        lines.append("")
    if summary["false_negatives"]:
        lines += ["missed, but labelled wrong", ""]
        for cid in summary["false_negatives"]:
            o = by_id[cid]
            lines.append(f"  {cid}  [{', '.join(sorted(o.labelled))}]  {o.why}")
        lines.append("")
    partial = [o for o in outcomes if o.truly_wrong and o.flagged and not (o.labelled - {"blind_spot"}) <= o.found]
    if partial:
        lines += ["caught, but not every labelled kind was found", ""]
        for o in partial:
            lines.append(f"  {o.case_id}  labelled {sorted(o.labelled)}, found {sorted(o.found)}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)

    cases, reviewed = load_cases(args.cases)
    tickets = {t.ticket_id: t for t in load_tickets(TICKETS_PATH)}
    customers = load_customers(CUSTOMERS_PATH)
    clause_text = {c.clause_id: c.text for c in load_clauses()}

    outcomes = [run_case(c, tickets, customers, clause_text) for c in cases]
    summary = summarise(outcomes)
    print(format_report(outcomes, summary, reviewed))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reviewed": reviewed,
            **summary,
            "per_case": [
                {"case_id": o.case_id, "labelled": sorted(o.labelled), "found": sorted(o.found), "details": o.details}
                for o in outcomes
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report written to {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
