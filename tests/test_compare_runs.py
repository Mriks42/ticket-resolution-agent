"""The cross-run comparison, on hand-built reports with known answers."""

from __future__ import annotations

import json

from evals.compare_runs import main


def report(path, outcomes: dict[str, str], resolution: float):
    gold = {"T-1": "resolve", "T-2": "escalate"}
    path.write_text(json.dumps({
        "metrics": [{"name": "resolution_rate", "value": resolution}],
        "per_ticket": [
            {"ticket_id": t, "gold_outcome": gold[t], "agent_outcome": o, "run_id": f"{path.stem}-{t}"}
            for t, o in outcomes.items()
        ],
    }), encoding="utf-8")
    return f"{path}"


def test_means_across_runs_and_counts_flips_within_a_configuration(tmp_path, capsys) -> None:
    a1 = report(tmp_path / "a1.json", {"T-1": "resolve", "T-2": "escalate"}, 0.5)
    a2 = report(tmp_path / "a2.json", {"T-1": "escalate", "T-2": "escalate"}, 0.0)
    b1 = report(tmp_path / "b1.json", {"T-1": "resolve", "T-2": "pending_approval"}, 0.5)

    assert main([f"a={a1}", f"a={a2}", f"b={b1}"]) == 0
    out = capsys.readouterr().out

    assert "mean 0.250  range 0.000-0.500" in out
    assert "T-1    R    RE*" in out
    assert "a          1/2" in out
    assert "b          0/2" in out
