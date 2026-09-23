from __future__ import annotations

from pathlib import Path

from evals.agent_scoring import evaluate_cases, load_cases, score_case


def test_scripted_agent_cases_pass(tmp_path: Path):
    rows = evaluate_cases(load_cases(), db_path=tmp_path / "receipts.db")
    assert len(rows) == 5
    assert all(row["ok"] for row in rows)


def test_score_case_rejects_a_mutation():
    result = {
        "stopped_reason": "needs_approval",
        "steps": [{"tool": "mark_used", "status": "needs_approval"}],
        "mutated": True,
    }
    expect = {
        "stopped_reason": "needs_approval",
        "first_tool": "mark_used",
        "first_status": "needs_approval",
        "mutated": False,
    }
    scored = score_case(result, expect)
    assert scored["mutated"] is False
    assert scored["ok"] is False
