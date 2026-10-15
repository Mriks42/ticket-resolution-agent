"""The guardrail harness, on a tiny case file with known answers."""

from __future__ import annotations

import json

import pytest

from evals.guardrail_eval import load_cases, main


def write(tmp_path, cases, reviewed=False):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"reviewed": reviewed, "cases": cases}), encoding="utf-8")
    return path


def case(case_id, reply, label, retrieved=("REF-001.1",)):
    return {
        "case_id": case_id, "ticket_id": "T-004", "retrieved": list(retrieved),
        "account": True, "refunds": [29], "reply": reply, "label": label, "why": "test",
    }


def test_counts_and_both_recalls(tmp_path, capsys) -> None:
    path = write(tmp_path, [
        case("C-1", "Your $29 charge is within the 30-day window (REF-001.1).", []),
        case("C-2", "You have 60 days to ask (REF-001.1).", [{"kind": "unsupported_fact"}]),
        case("C-3", "Annual plans are always refundable.", [{"kind": "blind_spot"}]),
    ])
    out_path = tmp_path / "report.json"
    assert main(["--cases", str(path), "--out", str(out_path)]) == 0

    report = json.loads(out_path.read_text())
    assert (report["true_positive"], report["false_positive"]) == (1, 0)
    assert (report["false_negative"], report["true_negative"]) == (1, 1)
    assert report["recall"] == 0.5
    assert report["recall_checkable"] == 1.0
    assert "UNREVIEWED" in capsys.readouterr().out


def test_an_unknown_clause_id_fails_loudly(tmp_path) -> None:
    path = write(tmp_path, [case("C-1", "x", [], retrieved=("REF-009.9",))])
    with pytest.raises(ValueError, match="REF-009.9"):
        load_cases(path)


def test_the_shipped_case_file_loads() -> None:
    cases, _ = load_cases()
    assert len(cases) >= 40
