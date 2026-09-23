from pathlib import Path
from types import SimpleNamespace

from app.services.agent import FALLBACK_TOKENS, AgentService
from tests.test_agent_loop import ScriptedProvider, _completion, _seed, _tool_call


def test_token_budget_keeps_the_finished_tool_step(tmp_path: Path):
    tools, _repo, _receipt_id = _seed(tmp_path)
    spent = _completion(tool_calls=[_tool_call("query_ledger", {"query_id": "count"})])
    spent.usage = SimpleNamespace(prompt_tokens=8, completion_tokens=4)
    provider = ScriptedProvider([spent, _completion(content="should not be called")])
    result = AgentService(provider, tools, token_budget=10).run("How many receipts?")
    assert result.stopped_reason == "fallback"
    assert result.answer == FALLBACK_TOKENS
    assert result.steps[0].tool == "query_ledger"
    assert result.steps[0].status == "ok"
    assert len(provider.calls) == 1
