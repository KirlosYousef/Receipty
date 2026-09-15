from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.services.extraction import ExtractionService
from evals.scoring import score_row


class ScriptedProvider:
    """Deterministic provider for CI evaluation cases."""

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = iter(responses)

    def complete(
        self, messages: list[dict[str, Any]], *, request_id: str | None = None
    ) -> Any:
        del messages, request_id
        content = json.dumps(next(self._responses))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def evaluate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    provider = ScriptedProvider([case["provider_response"] for case in cases])
    service = ExtractionService(provider)
    rows: list[dict[str, Any]] = []

    for case in cases:
        prediction = service.extract_from_image(
            case["input"].encode("utf-8"),
            "image/jpeg",
        )
        rows.append({"name": case["name"], **score_row(prediction, case["gold"])})

    return rows


def assert_no_hallucinations(rows: list[dict[str, Any]]) -> None:
    failures = [
        row["name"]
        for row in rows
        if row["hallucinated_total"] or row["hallucinated_date"]
    ]
    assert not failures, f"invented null-gold field(s): {', '.join(failures)}"
