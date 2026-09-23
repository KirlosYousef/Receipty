from pathlib import Path

from app.services.agent import FALLBACK_PROVIDER
from evals.agent_demo import render


def test_demo_shows_a_finished_task_and_a_kept_step(tmp_path: Path):
    text = render(tmp_path / "receipts.db")
    success, failure = text.split("failure\n", maxsplit=1)
    assert "stopped_reason: completed" in success
    assert "query_ledger ok" in success
    assert "stopped_reason: fallback" in failure
    assert "search_receipts ok" in failure
    assert FALLBACK_PROVIDER in failure
