from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ProviderError
from app.services.agent import FALLBACK_PROVIDER, FALLBACK_TIMEOUT, AgentService
from tests.test_agent_loop import ScriptedProvider, _completion, _seed, _tool_call


def test_provider_failure_keeps_earlier_tool_step(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)

    class FailOnSecond(ScriptedProvider):
        def complete(
            self, messages, *, request_id=None, response_format=None, tools=None
        ):
            if not self._completions:
                raise ProviderError("upstream down")
            return super().complete(
                messages,
                request_id=request_id,
                response_format=response_format,
                tools=tools,
            )

    provider = FailOnSecond(
        [_completion(tool_calls=[_tool_call("query_ledger", {"query_id": "count"})])]
    )
    result = AgentService(provider, tools).run("How many receipts?")
    assert result.stopped_reason == "fallback"
    assert result.answer == FALLBACK_PROVIDER
    assert result.steps[0].tool == "query_ledger"
    assert result.steps[0].status == "ok"


def test_time_budget_skips_the_model_call(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    times = iter([0.0, 5.0])
    provider = ScriptedProvider([_completion(content="should not be called")])
    result = AgentService(
        provider,
        tools,
        deadline_seconds=1,
        clock=lambda: next(times),
    ).run("Hello")
    assert result.stopped_reason == "fallback"
    assert result.answer == FALLBACK_TIMEOUT
    assert result.steps == []
    assert provider.calls == []
