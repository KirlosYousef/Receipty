from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from app.core.exceptions import ProviderError
from app.domain.schemas import AgentResponse
from app.services.agent import AgentService
from evals.agent_scoring import _ScriptedProvider, _seed


def _lines(title: str, question: str, result: AgentResponse) -> list[str]:
    steps = ", ".join(f"{step.tool} {step.status}" for step in result.steps) or "(none)"
    return [
        title,
        f"question: {question}",
        f"stopped_reason: {result.stopped_reason}",
        f"steps: {steps}",
        f"answer: {result.answer}",
        "",
    ]


class _FailAfterTurns(_ScriptedProvider):
    def complete(self, messages, *, request_id=None, response_format=None, tools=None):
        if not self._turns:
            raise ProviderError("upstream down")
        return super().complete(
            messages,
            request_id=request_id,
            response_format=response_format,
            tools=tools,
        )


def render(db_path: Path) -> str:
    tools, _repo, receipt_id = _seed(db_path)
    success_question = "How many receipts?"
    success = AgentService(
        _ScriptedProvider(
            [
                {
                    "tool_calls": [
                        {"name": "query_ledger", "arguments": {"query_id": "count"}}
                    ]
                },
                {"content": "There is 1 receipt."},
            ],
            receipt_id,
        ),
        tools,
    ).run(success_question)
    failure_question = "Find Taco Bell"
    failure = AgentService(
        _FailAfterTurns(
            [
                {
                    "tool_calls": [
                        {"name": "search_receipts", "arguments": {"q": "Taco Bell"}}
                    ]
                }
            ],
            receipt_id,
        ),
        tools,
    ).run(failure_question)
    lines = _lines("success", success_question, success) + _lines(
        "failure", failure_question, failure
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one finished agent task and one handled provider failure"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        text = render(Path(tmp) / "receipts.db")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
