from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.llm.provider import OpenRouterProvider
from app.observability.usage import UsageLogger
from app.services.extraction import ExtractionService
from evals.scoring import score_row, summarize_rows

FIXTURES = Path("evals/fixtures")
LABELS = Path("evals/labels.jsonl")


def mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def build_service() -> ExtractionService:
    settings = get_settings()
    provider = OpenRouterProvider(settings)
    usage = UsageLogger(settings.cost_log_path)
    return ExtractionService(provider, usage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Receipty image evals")
    parser.add_argument(
        "--json", type=Path, default=None, help="Write report JSON here"
    )
    args = parser.parse_args()

    labels = {
        r["file"]: r
        for line in LABELS.read_text().splitlines()
        if line.strip()
        for r in [json.loads(line)]
    }
    service = build_service()

    rows: list[dict] = []

    for name, gold in labels.items():
        path = FIXTURES / name
        if not path.exists():
            print("missing", name)
            rows.append({"file": name, "error": "missing_fixture", "ok": False})
            continue
        try:
            pred = service.extract_from_image(path.read_bytes(), mime_for(path))
        except Exception as e:
            print("FAIL", path.name, e)
            rows.append({"file": name, "error": str(e), "ok": False})
            continue

        scored = score_row(pred, gold)
        print(
            name,
            "\n",
            "receipt:",
            "PASS"
            if scored["receipt_ok"]
            else ("FAIL pred:", scored["pred_outcome"], "gold:", gold["is_receipt"]),
            "\n",
            "total:",
            "PASS"
            if scored["total_ok"]
            else ("FAIL pred:", scored["pred_total"], "gold:", scored["gold_total"]),
            "\n",
            "date:",
            "PASS"
            if scored["date_ok"]
            else ("FAIL pred:", scored["pred_date"], "gold:", scored["gold_date"]),
            "\n",
            "hallucinated_total:",
            scored["hallucinated_total"],
            "\n",
            "overall:",
            "PASS" if scored["ok"] else "FAIL",
            "\n",
        )
        rows.append({"file": name, **scored})

    summary = summarize_rows(rows)
    print(
        f"is_receipt {summary['is_receipt']}  "
        f"total {summary['total']}  "
        f"date {summary['date']}  "
        f"ok {summary['ok']}"
    )
    print(
        "hallucinated_total "
        f"{summary['hallucinated_total']['count']}/"
        f"{summary['hallucinated_total']['gold_null_total_cases']}  "
        "hallucinated_date "
        f"{summary['hallucinated_date']['count']}/"
        f"{summary['hallucinated_date']['gold_null_date_cases']}"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
