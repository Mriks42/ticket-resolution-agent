"""Measure `retrieve_policy` against the labels, with no agent loop involved.

    python -m evals.retrieval_baseline
    python -m evals.retrieval_baseline --backend lexical dense hybrid
    python -m evals.retrieval_baseline --backend dense --calibrate-floor
    python -m evals.retrieval_baseline --out evals/report/retrieval.json

Every number in ADR 0003 and ADR 0004 comes from here.

Retrieval quality is separable from agent quality, and measuring it on its own
is cheap: no LLM, no budgets, and for dense backends a few cents of embeddings
on the first run and nothing after (vectors are cached in Postgres). That makes
this the regression check for any change to the retrieval backend, and the
baseline a new backend has to beat.

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

**Recall with and without POL-ESC-001.** ADR 0003 classifies ESC-001's
approval bands as a gating rule decided by arithmetic, not a retrieval target,
so a backend that happens to surface ESC-001 more often is not a better
retriever for it. Both numbers are reported; backends are compared on the
second. One ticket is labeled with ESC-001 alone and drops out of it.

**Confidence intervals.** Per-ticket recall is bootstrapped over tickets with
the same seeded resampler `evals/run.py` uses. Backend differences are paired
— each resample draws the same tickets for both backends — which is the right
test when both ran on identical queries, and tighter than comparing two
independent intervals.

**Negative probes.** `evals/dataset/retrieval_probes.json` holds queries no
clause answers. The fraction a backend answers with an empty list is the only
direct measure of whether "nothing relevant" is expressible, which the
retrieval guardrail depends on. It is reported for lexical too: ADR 0003
assumed lexical has that property, and this checks rather than assumes it.

**The floor rule is fixed before the numbers.** `--calibrate-floor` picks the
highest floor on a 0.01 grid at which mean recall@k over the labeled tickets is
unchanged from the unfloored run — the most the floor can reject while costing
the labeled set nothing. Probe rejection is then reported as the consequence,
not optimised as the target: optimising rejection directly on 16 probes and 22
tickets would fit the floor to this set rather than calibrate it.

Dense and hybrid need Postgres (`docker compose up -d`) and `OPENAI_API_KEY`.
Without them the run fails loudly; there is no fallback to canned vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evals.loader import DATASET_DIR, POLICY_DIR, TICKETS_PATH, load_tickets
from evals.metrics import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci
from evals.schema import Outcome, Ticket
from src.config import load_dotenv
from src.tools.policy import Clause, load_clauses, load_policy_index
from src.tools.retrieval import Backend, HybridPolicyIndex, PolicySearch

#: Swept when choosing the default `limit` for `retrieve_policy`.
DEFAULT_K_SWEEP = (3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)

PROBES_PATH = DATASET_DIR / "retrieval_probes.json"

#: Candidate floors for `--calibrate-floor`.
FLOOR_GRID = tuple(round(i * 0.01, 2) for i in range(0, 81))

#: Tickets sitting on a deliberate corpus gap, per POLICY_NOTES.md. No clause
#: answers them; the nearest documents are labeled relevant because the agent
#: must read them to see that they stop short.
GAP_TICKETS = ("T-015", "T-017", "T-018")

#: Class-B per ADR 0003: decided by the approval gate, not by retrieval.
GATING_POLICY = "POL-ESC-001"

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
    #: Recall over relevant documents other than GATING_POLICY.
    per_ticket_excl_gate: dict[str, float] = field(default_factory=dict)

    @property
    def mean_recall_excl_gate(self) -> float:
        values = self.per_ticket_excl_gate.values()
        return sum(values) / len(values)

    @property
    def full_recall_excl_gate(self) -> int:
        return sum(1 for r in self.per_ticket_excl_gate.values() if r == 1.0)

    @property
    def mean_tokens(self) -> int:
        return round(self.mean_chars / CHARS_PER_TOKEN)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "mean_recall": round(self.mean_recall, 4),
            "full_recall_tickets": self.full_recall_tickets,
            "mean_recall_excl_esc_001": round(self.mean_recall_excl_gate, 4),
            "full_recall_tickets_excl_esc_001": (
                f"{self.full_recall_excl_gate}/{len(self.per_ticket_excl_gate)}"
            ),
            "esc_001_retrieved": f"{self.esc_hits}/{self.esc_total}",
            "mean_chars_returned": round(self.mean_chars, 1),
            "approx_mean_tokens": self.mean_tokens,
            "per_ticket_recall": {t: round(r, 4) for t, r in self.per_ticket.items()},
            "missing_documents": self.missing,
        }


@dataclass(frozen=True)
class Probe:
    probe_id: str
    tier: str
    query: str


@dataclass
class ProbeResult:
    empty: int
    total: int
    #: probe_id -> (top clause_id, its score) for every probe that returned something.
    leaks: dict[str, tuple[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "empty": self.empty,
            "total": self.total,
            "leaks": {p: {"clause_id": c, "score": round(s, 4)} for p, (c, s) in self.leaks.items()},
        }


@dataclass(frozen=True)
class FloorRow:
    floor: float
    mean_recall: float
    full_recall_tickets: int
    probes_empty: int
    mean_recall_excl_gate: float


@dataclass
class FloorCalibration:
    k: int
    unfloored_recall: float
    rows: list[FloorRow]
    chosen: FloorRow
    #: The same rule over recall excluding GATING_POLICY. Reported, not
    #: applied: the rule was fixed before the numbers and is not switched
    #: after seeing them.
    chosen_excl_gate: FloorRow
    top_similarity: dict[str, float]
    probe_top_similarity: dict[str, float]
    #: (similarity, ticket_id, policy_id, clause_id) of the weakest relevant
    #: documents in the unfloored top k — the hits that pin the floor.
    weakest_relevant: list[tuple[float, str, str, str]]

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "rule": "highest floor on a 0.01 grid with mean recall@k unchanged from unfloored",
            "unfloored_recall": round(self.unfloored_recall, 4),
            "chosen_floor": self.chosen.floor,
            "probes_empty_at_chosen": self.chosen.probes_empty,
            "variant_excl_esc_001_floor": self.chosen_excl_gate.floor,
            "probes_empty_at_variant": self.chosen_excl_gate.probes_empty,
            "weakest_relevant_hits": [
                {"similarity": round(s, 4), "ticket_id": t, "policy_id": p, "clause_id": c}
                for s, t, p, c in self.weakest_relevant
            ],
            "grid": [
                {
                    "floor": r.floor,
                    "mean_recall": round(r.mean_recall, 4),
                    "full_recall_tickets": r.full_recall_tickets,
                    "probes_empty": r.probes_empty,
                    "mean_recall_excl_esc_001": round(r.mean_recall_excl_gate, 4),
                }
                for r in self.rows
            ],
            "ticket_top_similarity": {t: round(s, 4) for t, s in self.top_similarity.items()},
            "probe_top_similarity": {p: round(s, 4) for p, s in self.probe_top_similarity.items()},
        }


# --- loading -------------------------------------------------------------------


def load_probes(path: Path = PROBES_PATH) -> tuple[list[Probe], bool]:
    """The probe set, and whether a human has reviewed it."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    probes = [Probe(p["probe_id"], p["tier"], p["query"]) for p in raw["probes"]]
    ids = [p.probe_id for p in probes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate probe ids")
    if any(not p.query.strip() for p in probes):
        raise ValueError(f"{path.name}: empty probe query")
    return probes, bool(raw.get("reviewed", False))


def build_backend(name: str, policies: Path, floor: float | None) -> PolicySearch:
    """An index for `name`. Dense ones are built from the environment directly
    rather than through `RetrievalConfig`, because calibration needs the one
    thing the runtime config forbids: a dense index with no floor."""
    if name == Backend.LEXICAL.value:
        return load_policy_index(policies)

    import psycopg

    from src.tools.dense import DensePolicyIndex
    from src.tools.embeddings import DEFAULT_EMBEDDING_MODEL, OpenAIEmbedder

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(f"{name}: DATABASE_URL is not set; run `docker compose up -d`")
    dense = DensePolicyIndex(
        load_clauses(policies),
        conn=psycopg.connect(url, autocommit=True),
        embedder=OpenAIEmbedder(
            model=os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
        ),
        floor=floor,
    )
    if name == Backend.DENSE.value:
        return dense
    return HybridPolicyIndex(load_policy_index(policies), dense)


def _dense_of(index: PolicySearch):
    return index.dense if isinstance(index, HybridPolicyIndex) else index


def describe(index: PolicySearch) -> dict:
    record = {"backend": index.backend, "clauses": len(index)}
    if index.backend != Backend.LEXICAL.value:
        dense = _dense_of(index)
        record |= {
            "embedding_model": dense.model,
            "floor": dense.floor,
            "clauses_embedded_this_run": dense.stats.clauses_embedded,
            "query_embeds_this_run": dense.stats.query_embeds,
            "query_cache_hits": dense.stats.query_cache_hits,
            "embedding_tokens_this_run": dense._embedder.tokens_used,
        }
    return record


# --- measuring ---------------------------------------------------------------


def _recall(hits: list[tuple[Clause, float]], wanted: set[str]) -> tuple[float, set[str]]:
    retrieved = {clause.policy_id for clause, _ in hits}
    return len(wanted & retrieved) / len(wanted), wanted - retrieved


def measure_k(index: PolicySearch, tickets: list[Ticket], k: int) -> KResult:
    recalls: dict[str, float] = {}
    recalls_excl_gate: dict[str, float] = {}
    missing: dict[str, list[str]] = {}
    chars: list[int] = []
    esc_hits = 0
    esc_total = 0

    for ticket in tickets:
        hits = index.search(ticket.body, k)
        chars.append(sum(len(clause.text) for clause, _ in hits))

        wanted = set(ticket.labels.relevant_policy_ids)
        recalls[ticket.ticket_id], absent = _recall(hits, wanted)
        if absent:
            missing[ticket.ticket_id] = sorted(absent)
        if wanted - {GATING_POLICY}:
            recalls_excl_gate[ticket.ticket_id] = _recall(hits, wanted - {GATING_POLICY})[0]

        if GATING_POLICY in wanted:
            esc_total += 1
            esc_hits += GATING_POLICY not in absent

    return KResult(
        k=k,
        mean_recall=sum(recalls.values()) / len(recalls),
        full_recall_tickets=sum(1 for r in recalls.values() if r == 1.0),
        esc_hits=esc_hits,
        esc_total=esc_total,
        mean_chars=sum(chars) / len(chars),
        per_ticket=recalls,
        missing=missing,
        per_ticket_excl_gate=recalls_excl_gate,
    )


def mean_ci(per_ticket: dict[str, float]) -> tuple[float, float]:
    """95% percentile-bootstrap interval on mean per-ticket recall."""
    low, high, _ = bootstrap_ci([(r, 1) for r in per_ticket.values()], DEFAULT_RESAMPLES, DEFAULT_SEED)
    return low, high


def paired_difference(
    candidate: dict[str, float], baseline: dict[str, float]
) -> tuple[float, float, float]:
    """Mean of candidate - baseline over shared tickets, with a 95% paired
    bootstrap interval. An interval spanning zero is not a measured win."""
    shared = sorted(candidate.keys() & baseline.keys())
    diffs = [(candidate[t] - baseline[t], 1) for t in shared]
    low, high, _ = bootstrap_ci(diffs, DEFAULT_RESAMPLES, DEFAULT_SEED)
    return sum(d for d, _ in diffs) / len(diffs), low, high


def measure_probes(index: PolicySearch, probes: list[Probe], k: int) -> ProbeResult:
    result = ProbeResult(empty=0, total=len(probes))
    for probe in probes:
        hits = index.search(probe.query, k)
        if hits:
            result.leaks[probe.probe_id] = (hits[0][0].clause_id, hits[0][1])
        else:
            result.empty += 1
    return result


def calibrate_floor(
    index: PolicySearch, tickets: list[Ticket], probes: list[Probe], k: int
) -> FloorCalibration:
    """Apply the floor rule in the module docstring.

    The unfloored hits are fetched once and the floor applied in Python. That
    is exact, not an approximation: hits come back best first, so a floored
    search at `k` is the unfloored top `k` with the tail below the floor cut.
    """
    dense = _dense_of(index)
    saved, dense.floor = dense.floor, None
    try:
        ticket_hits = {t.ticket_id: dense.search(t.body, k) for t in tickets}
        probe_top = {p.probe_id: dense.search(p.query, 1)[0][1] for p in probes}
    finally:
        dense.floor = saved

    wanted = {t.ticket_id: set(t.labels.relevant_policy_ids) for t in tickets}

    def recall_at(floor: float | None, exclude: set[str] = frozenset()) -> list[float]:
        return [
            _recall(
                [h for h in ticket_hits[tid] if floor is None or h[1] >= floor],
                wanted[tid] - exclude,
            )[0]
            for tid in ticket_hits
            if wanted[tid] - exclude
        ]

    unfloored = statistics.fmean(recall_at(None))
    unfloored_excl = statistics.fmean(recall_at(None, {GATING_POLICY}))
    rows = []
    for floor in FLOOR_GRID:
        recalls = recall_at(floor)
        rows.append(
            FloorRow(
                floor=floor,
                mean_recall=statistics.fmean(recalls),
                full_recall_tickets=sum(1 for r in recalls if r == 1.0),
                probes_empty=sum(1 for s in probe_top.values() if s < floor),
                mean_recall_excl_gate=statistics.fmean(recall_at(floor, {GATING_POLICY})),
            )
        )
    # Recall can only fall as the floor rises, so the lossless floors are a
    # prefix of the grid and the rule picks its last element.
    chosen = [r for r in rows if r.mean_recall >= unfloored - 1e-12][-1]
    chosen_excl = [r for r in rows if r.mean_recall_excl_gate >= unfloored_excl - 1e-12][-1]

    weakest = []
    for tid, hits in ticket_hits.items():
        best: dict[str, tuple[float, str]] = {}
        for clause, score in hits:
            best.setdefault(clause.policy_id, (score, clause.clause_id))
        for pid in wanted[tid] & best.keys():
            weakest.append((best[pid][0], tid, pid, best[pid][1]))
    weakest.sort()

    return FloorCalibration(
        k=k,
        unfloored_recall=unfloored,
        rows=rows,
        chosen=chosen,
        chosen_excl_gate=chosen_excl,
        weakest_relevant=weakest[:5],
        top_similarity={tid: hits[0][1] for tid, hits in ticket_hits.items()},
        probe_top_similarity=probe_top,
    )


def clause_rank(index: PolicySearch, query: str, clause_id: str) -> int | None:
    """1-based rank of a clause for a query, or None if it never surfaces."""
    ranked = [c.clause_id for c, _ in index.search(query, len(index))]
    return ranked.index(clause_id) + 1 if clause_id in ranked else None


def document_rank(
    index: PolicySearch, query: str, policy_id: str
) -> tuple[int | None, int]:
    """1-based rank of a document's best-scoring clause, and how many clauses
    the backend returned at all.

    The second number is what makes "ranked last" checkable. Lexical returns
    only clauses sharing a term with the query; a floored dense backend only
    clauses above the floor; an unfloored one returns every clause.
    """
    ranked = index.search(query, len(index))
    for position, (clause, _) in enumerate(ranked, 1):
        if clause.policy_id == policy_id:
            return position, len(ranked)
    return None, len(ranked)


# --- reporting ---------------------------------------------------------------


def _backend_title(index: PolicySearch) -> str:
    if index.backend == Backend.LEXICAL.value:
        return "lexical TF-IDF"
    dense = _dense_of(index)
    floor = "UNFLOORED - calibration only, not a deployable config" if dense.floor is None else f"floor {dense.floor}"
    kind = "dense" if index.backend == Backend.DENSE.value else "hybrid RRF(lexical, dense)"
    return f"{kind}, {dense.model}, {floor}"


def format_report(
    index: PolicySearch,
    tickets: list[Ticket],
    results: list[KResult],
    default_k: int,
    probes: ProbeResult | None = None,
    probes_reviewed: bool = True,
) -> str:
    lines = [
        "",
        f"retrieval baseline - {_backend_title(index)}, ticket body as query",
        f"corpus     {len(index)} clauses",
        f"tickets    {len(tickets)}",
        "recall is over relevant_policy_ids, clauses rolled up to documents",
        "",
        f"{'k':>4}{'recall':>9}{'full':>8}{'excl ESC':>10}{'full':>8}"
        f"{'ESC-001':>10}{'chars':>9}{'~tokens':>9}",
        "-" * 67,
    ]
    for r in results:
        marker = "  <- default" if r.k == default_k else ""
        n_excl = len(r.per_ticket_excl_gate)
        lines.append(
            f"{r.k:>4}{r.mean_recall:>9.3f}{r.full_recall_tickets:>4}/{len(tickets):<3}"
            f"{r.mean_recall_excl_gate:>10.3f}{r.full_recall_excl_gate:>4}/{n_excl:<3}"
            f"{r.esc_hits:>7}/{r.esc_total:<2}{r.mean_chars:>9.0f}{r.mean_tokens:>9}"
            f"{marker}"
        )

    lines += [
        "",
        "  recall   mean fraction of a ticket's relevant documents retrieved",
        "  full     tickets where every relevant document was retrieved",
        "  excl ESC recall over relevant documents other than POL-ESC-001, the",
        "           class-B gating policy (ADR 0003); tickets labeled with it alone drop out",
        "  ESC-001  of the tickets POL-ESC-001 is relevant to, how many retrieved it",
        "  chars    mean characters of passage text returned per ticket",
        "  ~tokens  chars/4, an approximation for context-budget purposes only",
        "",
    ]

    at_default = next(r for r in results if r.k == default_k)
    lo, hi = mean_ci(at_default.per_ticket)
    lo_x, hi_x = mean_ci(at_default.per_ticket_excl_gate)
    lines += [
        f"95% bootstrap CI at k={default_k}, over tickets",
        "",
        f"  recall           {at_default.mean_recall:.3f}  [{lo:.3f}, {hi:.3f}]",
        f"  recall excl ESC  {at_default.mean_recall_excl_gate:.3f}  [{lo_x:.3f}, {hi_x:.3f}]",
        "",
    ]
    worst = sorted(at_default.per_ticket.items(), key=lambda kv: kv[1])[:8]
    lines += [f"worst tickets at k={default_k}", ""]
    for ticket_id, recall in worst:
        gap = ", ".join(at_default.missing.get(ticket_id, [])) or "-"
        lines.append(f"  {ticket_id}  recall={recall:.2f}  missing: {gap}")

    lines += ["", "POL-ESC-001 rank on the pending_approval tickets", ""]
    for ticket in tickets:
        if ticket.labels.outcome is Outcome.PENDING_APPROVAL:
            rank, returned = document_rank(index, ticket.body, "POL-ESC-001")
            if rank is None:
                lines.append(f"  {ticket.ticket_id}  never retrieved")
                continue
            last = "  (last clause returned)" if rank == returned else ""
            lines.append(
                f"  {ticket.ticket_id}  rank {rank} of {returned} clauses returned{last}"
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

    if probes is not None:
        lines += [
            "",
            f"negative probes at k={default_k} - queries no clause answers",
            *([] if probes_reviewed else ["  ** probe set UNREVIEWED: these numbers are provisional **"]),
            "",
            f"  empty result  {probes.empty}/{probes.total}",
        ]
        for probe_id, (clause_id, score) in probes.leaks.items():
            lines.append(f"  {probe_id}  returned {clause_id} first (score {score:.3f})")

    lines.append("")
    return "\n".join(lines)


def format_calibration(cal: FloorCalibration, tickets: list[Ticket], probes_reviewed: bool) -> str:
    n = len(tickets)
    lines = [
        "",
        f"dense floor calibration at k={cal.k}",
        "rule: highest floor with mean recall unchanged from unfloored "
        f"({cal.unfloored_recall:.3f})",
        *(
            []
            if probes_reviewed
            else [
                "** probe set UNREVIEWED: the probes column is provisional. The chosen",
                "   floor is not - the rule reads ticket recall only, never the probes. **",
            ]
        ),
        "",
        f"{'floor':>7}{'recall':>9}{'full':>8}{'probes empty':>15}",
        "-" * 39,
    ]
    shown = {round(f * 0.05, 2) for f in range(0, 17)} | {cal.chosen.floor, cal.chosen_excl_gate.floor}
    total_probes = len(cal.probe_top_similarity)
    for row in cal.rows:
        if row.floor not in shown:
            continue
        marker = "  <- chosen" if row is cal.chosen else ""
        if row is cal.chosen_excl_gate:
            marker += "  <- variant excl ESC (reported, not applied)"
        lines.append(
            f"{row.floor:>7.2f}{row.mean_recall:>9.3f}{row.full_recall_tickets:>4}/{n:<3}"
            f"{row.probes_empty:>10}/{total_probes:<4}{marker}"
        )

    lines += ["", "weakest relevant documents in the unfloored top k - these pin the floor", ""]
    for sim, tid, pid, cid in cal.weakest_relevant:
        gate = "  (class-B gating policy)" if pid == GATING_POLICY else ""
        lines.append(f"  {sim:.4f}  {tid}  {pid:<12} via {cid}{gate}")

    ticket_tops = sorted(cal.top_similarity.items(), key=lambda kv: kv[1])
    probe_tops = sorted(cal.probe_top_similarity.items(), key=lambda kv: -kv[1])
    lines += [
        "",
        "top-1 similarity, tickets (lowest first) - gap tickets marked",
        "",
    ]
    for tid, sim in ticket_tops[:8]:
        mark = "  <- gap" if tid in GAP_TICKETS else ""
        lines.append(f"  {tid}  {sim:.3f}{mark}")
    others = [s for t, s in cal.top_similarity.items() if t not in GAP_TICKETS]
    for tid in GAP_TICKETS:
        if tid in cal.top_similarity:
            below = sum(1 for s in others if s < cal.top_similarity[tid])
            lines.append(
                f"  {tid} (gap) top-1 {cal.top_similarity[tid]:.3f}, above "
                f"{below}/{len(others)} covered tickets"
            )
    lines += ["", "top-1 similarity, probes (highest first)", ""]
    for pid, sim in probe_tops[:8]:
        lines.append(f"  {pid}  {sim:.3f}")
    lines.append("")
    return "\n".join(lines)


def format_comparison(at_default: dict[str, KResult], k: int) -> tuple[str, dict]:
    """Each backend against lexical, paired over tickets, at the default k."""
    base = at_default[Backend.LEXICAL.value]
    lines = [
        "",
        f"paired difference vs lexical at k={k}, 95% bootstrap CI",
        "an interval that spans 0.000 is not a measured difference",
        "",
    ]
    record: dict = {}
    for name, result in at_default.items():
        if name == Backend.LEXICAL.value:
            continue
        record[name] = {}
        for label, cand, ref in (
            ("recall", result.per_ticket, base.per_ticket),
            ("recall excl ESC", result.per_ticket_excl_gate, base.per_ticket_excl_gate),
        ):
            mean, low, high = paired_difference(cand, ref)
            spans = "  spans zero" if low <= 0.0 <= high else ""
            lines.append(f"  {name:<7}{label:<17}{mean:+.3f}  [{low:+.3f}, {high:+.3f}]{spans}")
            record[name][label.replace(" ", "_")] = {
                "mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]
            }
    lines.append("")
    return "\n".join(lines), record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=TICKETS_PATH)
    parser.add_argument("--policies", type=Path, default=POLICY_DIR)
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument(
        "--backend", nargs="+", default=[Backend.LEXICAL.value],
        choices=[b.value for b in Backend],
    )
    parser.add_argument(
        "--floor", type=float, default=None,
        help="Dense similarity floor. Defaults to RETRIEVAL_DENSE_FLOOR; "
        "unset means unfloored, which is reported as calibration-only.",
    )
    parser.add_argument(
        "--calibrate-floor", action="store_true",
        help="Sweep the dense floor and apply the selection rule.",
    )
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
    load_dotenv()

    if args.default_k is None:
        from src.tools.catalog import DEFAULT_RETRIEVAL_LIMIT

        args.default_k = DEFAULT_RETRIEVAL_LIMIT

    if args.floor is None:
        raw = os.environ.get("RETRIEVAL_DENSE_FLOOR", "").strip()
        args.floor = float(raw) if raw else None

    if args.calibrate_floor and Backend.DENSE.value not in args.backend:
        parser.error("--calibrate-floor needs the dense backend in --backend")

    tickets = load_tickets(args.dataset, args.policies)
    probes, probes_reviewed = load_probes(args.probes)
    ks = sorted(set(args.k) | {args.default_k})

    payload: dict = {
        "tickets": len(tickets),
        "default_k": args.default_k,
        "rollup": "clause ids rolled up to policy_id; labels are document-level",
        "probes_reviewed": probes_reviewed,
        "backends": [],
    }

    at_default: dict[str, KResult] = {}
    for name in args.backend:
        index = build_backend(name, args.policies, args.floor)
        results = [measure_k(index, tickets, k) for k in ks]
        at_default[name] = next(r for r in results if r.k == args.default_k)
        probe_result = measure_probes(index, probes, args.default_k)
        print(format_report(index, tickets, results, args.default_k, probe_result, probes_reviewed))

        record = {
            **describe(index),
            "sweep": [r.as_dict() for r in results],
            "probes": probe_result.as_dict(),
        }
        if args.calibrate_floor and name == Backend.DENSE.value:
            cal = calibrate_floor(index, tickets, probes, args.default_k)
            print(format_calibration(cal, tickets, probes_reviewed))
            record["floor_calibration"] = cal.as_dict()
        payload["backends"].append(record | describe(index))

    if Backend.LEXICAL.value in at_default and len(at_default) > 1:
        comparison = format_comparison(at_default, args.default_k)
        print(comparison[0])
        payload["paired_vs_lexical"] = comparison[1]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"report written to {args.out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
