from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

COMPATIBILITY_KEYS = (
    "model",
    "temperature",
    "seed",
    "prompt_hash",
    "labels_path",
    "fixtures_path",
)
BOOLEAN_METRICS = ("receipt_ok", "total_ok", "date_ok", "merchant_ok", "currency_ok")


class ReportComparisonError(ValueError):
    """Reports cannot support a meaningful repeated-run comparison."""


def compare_reports(
    reports: list[dict[str, Any]], *, allow_config_differences: bool = False
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ReportComparisonError("at least two reports are required")

    if allow_config_differences:
        _validate_shared_dataset(reports)
    else:
        _validate_compatible_metadata(reports)
    rows_by_file = [_rows_by_file(report) for report in reports]
    expected_files = set(rows_by_file[0])
    if any(set(rows) != expected_files for rows in rows_by_file[1:]):
        raise ReportComparisonError("fixture sets differ")

    changed = sorted(
        file
        for file in expected_files
        if len({_prediction_signature(rows[file]) for rows in rows_by_file}) > 1
    )
    return {
        "runs": len(reports),
        "configurations": [_configuration(report) for report in reports],
        "prediction_variance": {
            "cases": len(expected_files),
            "changed_cases": len(changed),
            "rate": _ratio(len(changed), len(expected_files)),
            "files": changed,
        },
        "metrics": {
            metric: _metric_variance(
                [
                    sum(1 for row in rows.values() if row.get(metric) is True)
                    for rows in rows_by_file
                ]
            )
            for metric in BOOLEAN_METRICS
        }
        | {
            "latency_ms": _metric_variance(
                [
                    _mean(
                        [
                            float(row["latency_ms"])
                            for row in rows.values()
                            if row.get("latency_ms") is not None
                        ]
                    )
                    for rows in rows_by_file
                ]
            )
        },
    }


def _validate_compatible_metadata(reports: list[dict[str, Any]]) -> None:
    baseline = reports[0].get("metadata", {})
    for report in reports[1:]:
        metadata = report.get("metadata", {})
        for key in COMPATIBILITY_KEYS:
            if metadata.get(key) != baseline.get(key):
                raise ReportComparisonError(f"metadata mismatch: {key}")


def _validate_shared_dataset(reports: list[dict[str, Any]]) -> None:
    baseline = reports[0].get("metadata", {})
    for report in reports[1:]:
        metadata = report.get("metadata", {})
        for key in ("labels_path", "fixtures_path"):
            if metadata.get(key) != baseline.get(key):
                raise ReportComparisonError(f"metadata mismatch: {key}")


def _configuration(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    return {
        key: metadata.get(key)
        for key in ("model", "temperature", "seed", "prompt_version", "prompt_hash")
    }


def _rows_by_file(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ReportComparisonError("report rows must be a list")
    by_file = {row.get("file"): row for row in rows}
    if None in by_file or len(by_file) != len(rows):
        raise ReportComparisonError("report rows must have unique file names")
    return by_file


def _prediction_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row.get(key)
        for key in (
            "pred_outcome",
            "pred_merchant",
            "pred_total",
            "pred_currency",
            "pred_date",
            "pred_tax",
        )
    )


def _metric_variance(values: list[float | int | None]) -> dict[str, Any]:
    available = [value for value in values if value is not None]
    return {
        "per_run": values,
        "min": min(available) if available else None,
        "max": max(available) if available else None,
        "mean": _mean(available),
        "population_stdev": statistics.pstdev(available) if available else None,
    }


def _mean(values: list[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare compatible evaluation reports"
    )
    parser.add_argument(
        "reports", type=Path, nargs="+", help="Two or more report JSON files"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write comparison JSON here"
    )
    parser.add_argument(
        "--allow-config-differences",
        action="store_true",
        help="Compare prompt/model experiments sharing the same labels and fixtures",
    )
    args = parser.parse_args(argv)
    try:
        result = compare_reports(
            [json.loads(path.read_text()) for path in args.reports],
            allow_config_differences=args.allow_config_differences,
        )
    except (OSError, json.JSONDecodeError, ReportComparisonError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
