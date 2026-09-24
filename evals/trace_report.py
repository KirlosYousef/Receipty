from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(record["duration_ms"]) for record in records]
    errors = sum(1 for record in records if record.get("error.type"))
    cost = sum(float(record.get("gen_ai.usage.cost") or 0) for record in records)
    count = len(records)
    if count == 0:
        return {
            "spans": 0,
            "errors": 0,
            "error_rate": None,
            "p50_ms": None,
            "p95_ms": None,
            "cost_usd_per_100": None,
        }
    return {
        "spans": count,
        "errors": errors,
        "error_rate": round(errors / count, 4),
        "p50_ms": round(_percentile(durations, 0.50), 3),
        "p95_ms": round(_percentile(durations, 0.95), 3),
        "cost_usd_per_100": round(cost / count * 100, 6),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize logs/traces.jsonl")
    parser.add_argument("--path", type=Path, default=Path("logs/traces.jsonl"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(load_records(args.path))
    text = json.dumps(summary, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
