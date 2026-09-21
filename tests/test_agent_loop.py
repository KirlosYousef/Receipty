from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.main import create_app
from app.repository.receipts import SqliteReceiptRepository
from app.services.agent import PENDING_ANSWER, AgentService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository
from app.services.tools import TOOL_DEFINITIONS, AgentTools


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion(*, content: str | None = None, tool_calls: list | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


class ScriptedProvider:
    def __init__(self, completions: list[Any]):
        self._completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "request_id": request_id,
                "response_format": response_format,
            }
        )
        if not self._completions:
            raise AssertionError("unexpected complete() call")
        return self._completions.pop(0)

    def close(self) -> None:
        pass


def _seed(tmp_path: Path) -> tuple[AgentTools, SqliteReceiptRepository, int]:
    db_path = tmp_path / "receipts.db"
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
    indexer = IndexingService(repo, HashEmbeddingProvider())
    indexer.index_receipt(receipt_id, extract)
    tools = AgentTools(
        repo,
        RetrievalService(SqliteRetrievalRepository(db_path), HashEmbeddingProvider()),
    )
    return tools, repo, receipt_id


def test_query_ledger_then_final_answer(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[
                    _tool_call(
                        "query_ledger",
                        {"query_id": "sum_total", "merchant": "Taco Bell"},
                    )
                ]
            ),
            _completion(content="Taco Bell totals 7.61 USD."),
        ]
    )
    result = AgentService(provider, tools).run("How much at Taco Bell?")
    assert result.stopped_reason == "completed"
    assert result.pending_mutation is None
    assert "7.61" in result.answer
    assert result.steps[0].tool == "query_ledger"
    assert result.steps[0].status == "ok"
    assert float(result.steps[0].result["amount"]) == pytest.approx(7.61)
    assert provider.calls[0]["tools"] == TOOL_DEFINITIONS
    assert provider.calls[0]["response_format"] is None


def test_write_tools_pause_without_mutating(tmp_path: Path):
    tools, repo, receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[_tool_call("mark_used", {"receipt_id": receipt_id})]
            ),
            _completion(content="should not be called"),
        ]
    )
    result = AgentService(provider, tools).run("Mark that receipt used.")
    assert result.stopped_reason == "needs_approval"
    assert result.answer == PENDING_ANSWER
    assert result.pending_mutation == {
        "tool": "mark_used",
        "args": {"receipt_id": receipt_id},
        "tool_call_id": "call_1",
    }
    assert int(repo.get(receipt_id)["used"]) == 0
    assert len(provider.calls) == 1


def test_invalid_write_args_do_not_pause(tmp_path: Path):
    tools, repo, receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[
                    _tool_call(
                        "mark_used",
                        {"receipt_id": receipt_id, "sql": "DROP TABLE receipts"},
                    )
                ]
            ),
            _completion(content="That write was rejected."),
        ]
    )
    result = AgentService(provider, tools).run("Mark used with extra fields.")
    assert result.stopped_reason == "completed"
    assert result.steps[0].status == "error"
    assert int(repo.get(receipt_id)["used"]) == 0
    assert "rejected" in result.answer


def test_unknown_tool_is_returned_as_error(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(tool_calls=[_tool_call("drop_table", {"name": "receipts"})]),
            _completion(content="I cannot run that tool."),
        ]
    )
    result = AgentService(provider, tools).run("Drop the table.")
    assert result.stopped_reason == "completed"
    assert result.steps[0].status == "error"
    assert "unknown tool" in result.steps[0].result["error"]


def test_malformed_tool_arguments_are_errors(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    bad = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="query_ledger", arguments="{not json"),
    )
    provider = ScriptedProvider(
        [
            _completion(tool_calls=[bad]),
            _completion(content="Those arguments were invalid."),
        ]
    )
    result = AgentService(provider, tools).run("Sum totals.")
    assert result.stopped_reason == "completed"
    assert result.steps[0].status == "error"
    assert "invalid tool arguments" in result.steps[0].result["error"]


def test_max_steps_stops_a_tool_loop(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[
                    _tool_call("search_receipts", {"q": "Taco Bell"}, call_id="c1")
                ]
            ),
            _completion(
                tool_calls=[
                    _tool_call("search_receipts", {"q": "Taco Bell"}, call_id="c2")
                ]
            ),
            _completion(content="should not be called"),
        ]
    )
    result = AgentService(provider, tools, max_steps=2).run("Find Taco Bell.")
    assert result.stopped_reason == "max_steps"
    assert result.pending_mutation is None
    assert len(result.steps) == 2
    assert len(provider.calls) == 2
    assert "2 tool rounds" in result.answer


def test_read_then_write_in_one_round_pauses_after_read(tmp_path: Path):
    tools, repo, receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[
                    _tool_call(
                        "query_ledger",
                        {"query_id": "count"},
                        call_id="c1",
                    ),
                    _tool_call(
                        "flag_for_review",
                        {"receipt_id": receipt_id},
                        call_id="c2",
                    ),
                ]
            ),
            _completion(content="should not be called"),
        ]
    )
    result = AgentService(provider, tools).run("Count then flag.")
    assert result.stopped_reason == "needs_approval"
    assert result.steps[0].status == "ok"
    assert result.steps[1].status == "needs_approval"
    row = repo.get(receipt_id)
    assert row["outcome"] == Outcome.success.value
    assert int(row["used"]) == 0
    assert len(provider.calls) == 1


def test_agent_http_runs_read_tools(tmp_path: Path):
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[
                    _tool_call(
                        "query_ledger",
                        {"query_id": "count", "merchant": "Taco Bell"},
                    )
                ]
            ),
            _completion(content="There is 1 Taco Bell receipt."),
        ]
    )
    settings = Settings(
        openrouter_api_key="test-key",
        db_path=tmp_path / "http.db",
        cost_log_path=tmp_path / "cost.jsonl",
        max_agent_steps=4,
    )
    app = create_app(
        settings=settings,
        provider_factory=lambda _: provider,
        embedding_factory=lambda _: HashEmbeddingProvider(),
    )
    with TestClient(app) as client:
        extract = ReceiptExtract(
            is_receipt=True,
            merchant="Taco Bell",
            total="7.61",
            currency="USD",
            date="2016-09-01",
            outcome=Outcome.success,
        )
        saved = app.state.repo.save(extract)
        app.state.indexer.index_receipt(saved, extract)
        empty = client.post("/v1/agent", json={"question": ""})
        assert empty.status_code == 422
        response = client.post(
            "/v1/agent",
            json={"question": "How many Taco Bell receipts?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["stopped_reason"] == "completed"
        assert body["steps"][0]["tool"] == "query_ledger"
        assert body["steps"][0]["result"]["n"] == 1
