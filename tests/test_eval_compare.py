import pytest

from evals.compare import ReportComparisonError, compare_reports


def report(*, rows, model="model-a", prompt_hash="prompt-1"):
    return {
        "metadata": {
            "model": model,
            "temperature": 0.0,
            "seed": 42,
            "prompt_hash": prompt_hash,
            "labels_path": "evals/labels.jsonl",
            "fixtures_path": "evals/fixtures",
        },
        "rows": rows,
    }


def row(file, total, latency_ms=100.0):
    return {
        "file": file,
        "receipt_ok": True,
        "total_ok": total is not None,
        "date_ok": True,
        "merchant_ok": True,
        "currency_ok": True,
        "pred_outcome": "success",
        "pred_merchant": "Shop",
        "pred_total": total,
        "pred_currency": "USD",
        "pred_date": "2026-01-01",
        "pred_tax": None,
        "latency_ms": latency_ms,
    }


def test_compare_reports_summarizes_output_and_metric_variance():
    comparison = compare_reports(
        [
            report(rows=[row("one.jpg", "10.00"), row("two.jpg", "20.00")]),
            report(
                rows=[row("one.jpg", "10.00", 120.0), row("two.jpg", "21.00", 80.0)]
            ),
        ]
    )

    assert comparison["runs"] == 2
    assert comparison["prediction_variance"] == {
        "cases": 2,
        "changed_cases": 1,
        "rate": 0.5,
        "files": ["two.jpg"],
    }
    assert comparison["metrics"]["total_ok"] == {
        "per_run": [2, 2],
        "min": 2,
        "max": 2,
        "mean": 2.0,
        "population_stdev": 0.0,
    }
    assert comparison["metrics"]["latency_ms"]["per_run"] == [100.0, 100.0]
    assert comparison["configurations"][0]["model"] == "model-a"


def test_compare_reports_rejects_incompatible_configuration():
    with pytest.raises(ReportComparisonError, match="metadata mismatch: model"):
        compare_reports(
            [
                report(rows=[row("one.jpg", "10.00")]),
                report(rows=[row("one.jpg", "10.00")], model="model-b"),
            ]
        )


def test_compare_reports_rejects_different_fixture_sets():
    with pytest.raises(ReportComparisonError, match="fixture sets differ"):
        compare_reports(
            [
                report(rows=[row("one.jpg", "10.00")]),
                report(rows=[row("two.jpg", "10.00")]),
            ]
        )


def test_compare_reports_allows_prompt_or_model_experiments_on_same_dataset():
    comparison = compare_reports(
        [
            report(rows=[row("one.jpg", "10.00")], model="model-a"),
            report(rows=[row("one.jpg", "10.00")], model="model-b"),
        ],
        allow_config_differences=True,
    )

    assert [config["model"] for config in comparison["configurations"]] == [
        "model-a",
        "model-b",
    ]
