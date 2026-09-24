import json
from pathlib import Path

from evals.trace_report import load_records, summarize


def test_summary_reports_latency_errors_and_cost_per_100():
    records = [
        {"duration_ms": 10, "gen_ai.usage.cost": 0.01},
        {"duration_ms": 20},
        {"duration_ms": 30, "error.type": "ProviderError", "gen_ai.usage.cost": 0.02},
        {"duration_ms": 40},
    ]
    summary = summarize(records)
    assert summary["spans"] == 4
    assert summary["errors"] == 1
    assert summary["error_rate"] == 0.25
    assert summary["p50_ms"] == 20
    assert summary["p95_ms"] == 40
    assert summary["cost_usd_per_100"] == 0.75


def test_missing_file_is_an_empty_summary(tmp_path: Path):
    assert summarize(load_records(tmp_path / "missing.jsonl"))["spans"] == 0


def test_load_records_reads_jsonl(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps({"duration_ms": 5}) + "\n")
    assert load_records(path) == [{"duration_ms": 5}]
