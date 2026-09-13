from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.llm.provider import OpenRouterProvider
from app.observability.usage import UsageLogger
from app.services.extraction import ExtractionService
from evals.scoring import score_row

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

    ok_receipt = ok_total = 0
    n = 0
    rows: list[dict] = []

    for name, gold in labels.items():
        path = FIXTURES / name
        if not path.exists():
            print("missing", name)
            continue
        n += 1
        try:
            pred = service.extract_from_image(path.read_bytes(), mime_for(path))
        except Exception as e:
            print("FAIL", path.name, e)
            rows.append({"file": name, "error": str(e), "ok": False})
            continue

        scored = score_row(pred, gold)
        ok_receipt += int(scored["receipt_ok"])
        ok_total += int(scored["total_ok"])
        flag = "OK" if scored["ok"] else "FAIL"
        print(
            flag,
            name,
            "pred",
            scored["pred_merchant"],
            scored["pred_total"],
            "gold",
            scored["gold_merchant"],
            scored["gold_total"],
        )
        rows.append({"file": name, **scored})

    summary = {
        "n": n,
        "is_receipt": f"{ok_receipt}/{n}",
        "total": f"{ok_total}/{n}",
        "is_receipt_correct": ok_receipt,
        "total_correct": ok_total,
    }
    print(f"is_receipt {ok_receipt}/{n}  total {ok_total}/{n}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
