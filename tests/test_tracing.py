import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderDeadlineExceeded
from app.llm.embeddings import HashEmbeddingProvider
from app.llm.provider import OpenRouterProvider
from app.observability.tracing import SpanRecorder, bind_request_id
from app.services.retrieval import RetrievalService
from app.services.tools import AgentTools, ToolError


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_chat_span_records_tokens_cost_and_request_id(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    completions = SimpleNamespace(
        create=lambda **_kwargs: SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, cost=0.02)
        )
    )
    provider = OpenRouterProvider(
        Settings(openrouter_api_key="test-key", model="test-model"),
        spans=SpanRecorder(path),
    )
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    provider.complete([{"role": "user", "content": "hi"}], request_id="req-1")

    record = _records(path)[0]
    assert record["name"] == "chat test-model"
    assert record["request_id"] == "req-1"
    assert record["gen_ai.operation.name"] == "chat"
    assert record["gen_ai.provider.name"] == "openrouter"
    assert record["gen_ai.request.model"] == "test-model"
    assert record["gen_ai.usage.input_tokens"] == 11
    assert record["gen_ai.usage.output_tokens"] == 4
    assert record["gen_ai.usage.cost"] == 0.02
    assert record["duration_ms"] >= 0


def test_chat_span_records_the_error_type(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    times = iter([0.0, 5.0])
    provider = OpenRouterProvider(
        Settings(openrouter_api_key="test-key", total_deadline_seconds=1),
        spans=SpanRecorder(path),
        clock=lambda: next(times),
    )
    with pytest.raises(ProviderDeadlineExceeded):
        provider.complete([{"role": "user", "content": "hi"}], request_id="req-2")
    record = _records(path)[0]
    assert record["error.type"] == "ProviderDeadlineExceeded"
    assert "gen_ai.usage.input_tokens" not in record


def test_retrieval_and_tool_spans_keep_the_request_id(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    spans = SpanRecorder(path)

    class KeywordRepo:
        def search_keyword(
            self, query: str, *, limit: int, kind: str | None
        ) -> list[dict]:
            del query, limit, kind
            return []

    retrieval = RetrievalService(KeywordRepo(), HashEmbeddingProvider(), spans=spans)
    tools = AgentTools(
        SimpleNamespace(ledger_count=lambda _merchant: {"count": 1}),
        retrieval,
        spans=spans,
    )

    with bind_request_id("req-3"):
        retrieval.search("taco", strategy="keyword")
        tools.dispatch("query_ledger", {"query_id": "count"})
        with pytest.raises(ToolError):
            tools.dispatch("drop_table", {})

    rows = _records(path)
    assert rows[0]["name"] == "retrieval"
    assert rows[0]["gen_ai.operation.name"] == "retrieval"
    assert rows[0]["receipty.retrieval.strategy"] == "keyword"
    assert rows[0]["request_id"] == "req-3"
    assert rows[1]["name"] == "execute_tool query_ledger"
    assert rows[1]["gen_ai.tool.name"] == "query_ledger"
    assert rows[1]["request_id"] == "req-3"
    assert rows[2]["error.type"] == "ToolError"
    assert "arguments" not in rows[1]
