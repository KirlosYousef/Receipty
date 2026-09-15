from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.domain.schemas import ReceiptExtract
from evals import runner
from evals.runner import EvaluationRunError, run_evaluation


class StubService:
    def __init__(self, result: ReceiptExtract | Exception):
        self._result = result

    def extract_from_image(self, image_bytes: bytes, mime: str) -> ReceiptExtract:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def write_label(path: Path, filename: str) -> None:
    path.write_text(
        json.dumps(
            {
                "file": filename,
                "is_receipt": True,
                "merchant": "Example",
                "total": 12.34,
                "currency": "USD",
                "date": "2026-09-15",
            }
        )
        + "\n"
    )


def test_missing_fixture_fails_evaluation(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    write_label(labels_path, "missing.jpg")

    with pytest.raises(EvaluationRunError, match="missing fixtures: missing.jpg"):
        run_evaluation(
            labels_path=labels_path,
            fixtures_path=tmp_path / "fixtures",
            service=StubService(ReceiptExtract(is_receipt=True)),
            metadata={"commit_sha": "test"},
        )


def test_provider_failure_fails_evaluation(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures"
    fixtures_path.mkdir()
    (fixtures_path / "receipt.jpg").write_bytes(b"fixture")
    labels_path = tmp_path / "labels.jsonl"
    write_label(labels_path, "receipt.jpg")

    with pytest.raises(EvaluationRunError, match="provider failures: receipt.jpg"):
        run_evaluation(
            labels_path=labels_path,
            fixtures_path=fixtures_path,
            service=StubService(RuntimeError("provider unavailable")),
            metadata={"commit_sha": "test"},
        )


def test_main_returns_non_zero_for_invalid_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    write_label(labels_path, "missing.jpg")
    monkeypatch.setattr(runner, "LABELS", labels_path)
    monkeypatch.setattr(runner, "FIXTURES", tmp_path / "fixtures")
    monkeypatch.setattr(
        runner,
        "build_service",
        lambda *_: (StubService(ReceiptExtract(is_receipt=True)), None, Settings()),
    )

    assert runner.main([]) == 1
    assert "EVAL INVALID: missing fixtures: missing.jpg" in capsys.readouterr().err


def test_report_contains_metadata_and_case_latency(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures"
    fixtures_path.mkdir()
    (fixtures_path / "receipt.jpg").write_bytes(b"fixture")
    labels_path = tmp_path / "labels.jsonl"
    write_label(labels_path, "receipt.jpg")

    report = run_evaluation(
        labels_path=labels_path,
        fixtures_path=fixtures_path,
        service=StubService(
            ReceiptExtract(
                is_receipt=True,
                merchant="Example",
                total="12.34",
                currency="USD",
                date="2026-09-15",
            )
        ),
        metadata={
            "created_at": "2026-09-15T00:00:00+00:00",
            "commit_sha": "abc123",
            "model": "test-model",
            "temperature": 0.0,
            "seed": 42,
            "prompt_version": "extraction-v1",
            "prompt_hash": "deadbeef",
        },
    )

    assert report["metadata"]["commit_sha"] == "abc123"
    assert report["metadata"]["prompt_hash"] == "deadbeef"
    assert report["summary"]["ok"] == "1/1"
    assert report["rows"][0]["latency_ms"] >= 0


def test_run_metadata_records_selected_prompt_version() -> None:
    metadata = runner.run_metadata(Settings(), "extraction-v2-evidence")

    assert metadata["prompt_version"] == "extraction-v2-evidence"
    assert metadata["prompt_hash"] != runner.run_metadata(Settings())["prompt_hash"]
