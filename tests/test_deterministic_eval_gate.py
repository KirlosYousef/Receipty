from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.ci_gate import assert_no_hallucinations, evaluate_cases

CASES_PATH = Path("evals/ci_cases.json")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def test_deterministic_cases_pass_the_production_extraction_path() -> None:
    rows = evaluate_cases(load_cases())

    assert len(rows) == 3
    assert all(row["ok"] for row in rows)
    assert_no_hallucinations(rows)


def test_invented_unreadable_total_fails_the_safety_gate() -> None:
    cases = load_cases()
    unreadable_total = next(
        case for case in cases if case["name"] == "unreadable_total"
    )
    unreadable_total["provider_response"]["total"] = "999.99"

    rows = evaluate_cases(cases)

    assert rows[1]["hallucinated_total"] is True
    with pytest.raises(AssertionError, match="unreadable_total"):
        assert_no_hallucinations(rows)
