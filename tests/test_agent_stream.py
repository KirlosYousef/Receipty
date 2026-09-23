from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.llm.embeddings import HashEmbeddingProvider
from app.main import create_app
from app.services.agent import AgentService
from tests.test_agent_loop import ScriptedProvider, _completion, _seed, _tool_call


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((name, json.loads(data)))
    return events


def test_stream_emits_tool_step_before_final_answer(tmp_path):
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
    events = list(AgentService(provider, tools).stream("How much at Taco Bell?"))
    assert events[0]["event"] == "step"
    assert events[0]["data"]["tool"] == "query_ledger"
    assert events[0]["data"]["status"] == "ok"
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["stopped_reason"] == "completed"
    assert "7.61" in events[-1]["data"]["answer"]


def test_stream_write_pauses_and_resume_still_applies(tmp_path):
    tools, repo, receipt_id = _seed(tmp_path)
    provider = ScriptedProvider(
        [
            _completion(
                tool_calls=[_tool_call("mark_used", {"receipt_id": receipt_id})]
            ),
            _completion(content="Marked as used."),
        ]
    )
    service = AgentService(provider, tools)
    events = list(service.stream("Mark that receipt used."))
    done = events[-1]["data"]
    assert done["stopped_reason"] == "needs_approval"
    assert int(repo.get(receipt_id)["used"]) == 0
    resumed = service.resume(done["thread_id"], approved=True)
    assert resumed.stopped_reason == "completed"
    assert int(repo.get(receipt_id)["used"]) == 1


def test_agent_stream_http_and_provider_error(tmp_path):
    provider = ScriptedProvider([])
    settings = Settings(
        openrouter_api_key="test-key",
        db_path=tmp_path / "sse.db",
        cost_log_path=tmp_path / "cost.jsonl",
    )
    app = create_app(
        settings=settings,
        provider_factory=lambda _: provider,
        embedding_factory=lambda _: HashEmbeddingProvider(),
    )
    with TestClient(app) as client:
        provider._completions = [
            _completion(
                tool_calls=[
                    _tool_call("query_ledger", {"query_id": "count"})
                ]
            ),
            _completion(content="There are 0 receipts."),
        ]
        with client.stream(
            "POST", "/v1/agent/stream", json={"question": "How many receipts?"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = _parse_sse("".join(response.iter_text()))
        assert events[0][0] == "step"
        assert events[-1][0] == "done"
        assert events[-1][1]["stopped_reason"] == "completed"

        def fail(*_args, **_kwargs):
            raise ProviderError("upstream down")

        provider.complete = fail
        with client.stream(
            "POST", "/v1/agent/stream", json={"question": "How many receipts?"}
        ) as response:
            error_events = _parse_sse("".join(response.iter_text()))
        assert error_events == [
            ("error", {"status": 502, "detail": "upstream down"})
        ]
