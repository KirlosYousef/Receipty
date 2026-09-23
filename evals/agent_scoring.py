from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.repository.receipts import SqliteReceiptRepository
from app.services.agent import AgentService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository
from app.services.tools import AgentTools

CASES_PATH = Path(__file__).with_name("agent_cases.jsonl")


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_case(result: dict[str, Any], expect: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "stopped_reason": result["stopped_reason"] == expect["stopped_reason"],
        "first_tool": (result["steps"][0]["tool"] if result["steps"] else None)
        == expect["first_tool"],
        "first_status": (result["steps"][0]["status"] if result["steps"] else None)
        == expect["first_status"],
        "mutated": result["mutated"] is expect["mutated"],
    }
    return {**checks, "ok": all(checks.values())}


class _ScriptedProvider:
    def __init__(self, turns: list[dict[str, Any]], receipt_id: int):
        self._turns = [_bind(turn, receipt_id) for turn in turns]

    def complete(self, messages, *, request_id=None, response_format=None, tools=None):
        del messages, request_id, response_format, tools
        if not self._turns:
            raise AssertionError("scripted provider ran out of turns")
        turn = self._turns.pop(0)
        tool_calls = None
        if "tool_calls" in turn:
            tool_calls = [
                SimpleNamespace(
                    id=f"call_{index}",
                    type="function",
                    function=SimpleNamespace(
                        name=call["name"],
                        arguments=json.dumps(call["arguments"]),
                    ),
                )
                for index, call in enumerate(turn["tool_calls"], start=1)
            ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=turn.get("content"),
                        tool_calls=tool_calls,
                    )
                )
            ]
        )

    def close(self) -> None:
        return None


def _bind(turn: dict[str, Any], receipt_id: int) -> dict[str, Any]:
    raw = json.dumps(turn).replace('"$seed"', str(receipt_id))
    return json.loads(raw)


def _seed(db_path: Path) -> tuple[AgentTools, SqliteReceiptRepository, int]:
    repo = SqliteReceiptRepository(db_path)
    repo.init_db()
    extract = ReceiptExtract(
        is_receipt=True,
        merchant="Taco Bell",
        total="7.61",
        currency="USD",
        date="2016-09-01",
        outcome=Outcome.success,
    )
    receipt_id = repo.save(extract)
    IndexingService(repo, HashEmbeddingProvider()).index_receipt(receipt_id, extract)
    tools = AgentTools(
        repo,
        RetrievalService(SqliteRetrievalRepository(db_path), HashEmbeddingProvider()),
    )
    return tools, repo, receipt_id


def evaluate_cases(
    cases: list[dict[str, Any]], *, db_path: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        tools, repo, receipt_id = _seed(db_path)
        before = int(repo.get(receipt_id)["used"])
        service = AgentService(
            _ScriptedProvider(case["turns"], receipt_id),
            tools,
            max_steps=int(case["max_steps"]),
        )
        result = service.run(case["question"])
        payload = {
            "stopped_reason": result.stopped_reason,
            "steps": [step.model_dump() for step in result.steps],
            "mutated": int(repo.get(receipt_id)["used"]) != before,
        }
        scored = score_case(payload, case["expect"])
        rows.append({"id": case["id"], **scored})
    return rows
