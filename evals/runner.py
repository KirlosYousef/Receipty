from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.core.config import Settings, get_settings
from app.llm.prompts import DEFAULT_PROMPT_VERSION, EXTRACTION_PROMPTS, prompt_for
from app.llm.provider import OpenRouterProvider
from app.observability.usage import UsageLogger
from app.services.extraction import ExtractionService
from evals.scoring import score_row, summarize_rows

FIXTURES = Path("evals/fixtures")
LABELS = Path("evals/labels.jsonl")


class ImageExtractionService(Protocol):
    def extract_from_image(self, image_bytes: bytes, mime: str) -> Any: ...


class EvaluationRunError(RuntimeError):
    """The report is incomplete and must not be used as an eval result."""


def mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


class EvaluationUsageLogger:
    """Persist normal usage logs and retain one serial evaluation call's usage."""

    def __init__(self, delegate: UsageLogger):
        self._delegate = delegate
        self._latest: dict[str, Any] | None = None

    def log(self, completion: Any, kind: str) -> None:
        self._delegate.log(completion, kind)
        usage = completion.usage
        extra = getattr(usage, "model_extra", None) or {}
        usd = getattr(usage, "cost", None)
        self._latest = {
            "model": getattr(completion, "model", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "usd": usd if usd is not None else extra.get("cost"),
        }

    def take_latest(self) -> dict[str, Any]:
        latest = self._latest or {}
        self._latest = None
        return latest


def build_service(
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> tuple[ExtractionService, EvaluationUsageLogger, Settings]:
    settings = get_settings()
    provider = OpenRouterProvider(settings)
    usage = EvaluationUsageLogger(UsageLogger(settings.cost_log_path))
    return (
        ExtractionService(provider, usage, prompt=prompt_for(prompt_version)),
        usage,
        settings,
    )


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["file"]: row
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def run_metadata(
    settings: Settings, prompt_version: str = DEFAULT_PROMPT_VERSION
) -> dict[str, Any]:
    prompt = prompt_for(prompt_version)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_sha(),
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "prompt_version": prompt_version,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "labels_path": str(LABELS),
        "fixtures_path": str(FIXTURES),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None


def run_evaluation(
    *,
    labels_path: Path,
    fixtures_path: Path,
    service: ImageExtractionService,
    metadata: dict[str, Any],
    usage_logger: EvaluationUsageLogger | None = None,
) -> dict[str, Any]:
    labels = load_labels(labels_path)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []

    for name, gold in labels.items():
        path = fixtures_path / name
        if not path.exists():
            missing.append(name)
            continue

        started = time.perf_counter()
        try:
            pred = service.extract_from_image(path.read_bytes(), mime_for(path))
        except Exception as exc:
            failed.append(name)
            rows.append({"file": name, "error": str(exc), "ok": False})
            continue

        usage = usage_logger.take_latest() if usage_logger is not None else {}
        scored = score_row(pred, gold)
        scored.update(
            {
                "file": name,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "model": usage.get("model"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": _total_tokens(usage),
                "usd": usage.get("usd"),
            }
        )
        rows.append(scored)

    errors: list[str] = []
    if missing:
        errors.append(f"missing fixtures: {', '.join(missing)}")
    if failed:
        errors.append(f"provider failures: {', '.join(failed)}")
    if errors:
        raise EvaluationRunError("; ".join(errors))

    return {"metadata": metadata, "summary": summarize_rows(rows), "rows": rows}


def _total_tokens(usage: dict[str, Any]) -> int | None:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    return int(prompt) + int(completion)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Receipty image evals")
    parser.add_argument(
        "--json", type=Path, default=None, help="Write report JSON here"
    )
    parser.add_argument(
        "--prompt-version",
        choices=sorted(EXTRACTION_PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
        help="Versioned extraction prompt to evaluate",
    )
    args = parser.parse_args(argv)
    service, usage_logger, settings = build_service(args.prompt_version)
    try:
        report = run_evaluation(
            labels_path=LABELS,
            fixtures_path=FIXTURES,
            service=service,
            usage_logger=usage_logger,
            metadata=run_metadata(settings, args.prompt_version),
        )
    except EvaluationRunError as exc:
        print(f"EVAL INVALID: {exc}", file=sys.stderr)
        return 1

    summary = report["summary"]
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
    latency = summary["latency_ms"]
    print(
        f"latency_ms p50={latency['p50']} p95={latency['p95']} (n={latency['count']})"
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
