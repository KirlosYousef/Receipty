from __future__ import annotations

import json
import logging
from typing import Any

from app.domain.schemas import AgentResponse, AgentStep
from app.llm.prompts import AGENT_PROMPT
from app.llm.provider import LLMProvider
from app.services.tools import (
    TOOL_DEFINITIONS,
    WRITE_TOOLS,
    AgentTools,
    ReceiptIdArgs,
    ToolError,
    parse_args,
)

log = logging.getLogger(__name__)

MAX_STEPS_ANSWER = "Stopped after {n} tool rounds without a final answer."
PENDING_ANSWER = "This write needs approval before it runs."
EMPTY_ANSWER = "I could not produce an answer."


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError(f"invalid tool arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolError("tool arguments must be a JSON object")
    return parsed


def _assistant_tool_message(message: Any) -> dict[str, Any]:
    calls = []
    for call in message.tool_calls:
        arguments = call.function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": calls,
    }


def _tool_message(call_id: str, payload: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, default=str),
    }


class AgentService:
    def __init__(
        self,
        provider: LLMProvider,
        tools: AgentTools,
        *,
        max_steps: int = 8,
        prompt: str = AGENT_PROMPT,
    ):
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._prompt = prompt

    def run(self, question: str, *, request_id: str | None = None) -> AgentResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []

        for _round in range(self._max_steps):
            completion = self._provider.complete(
                messages,
                request_id=request_id,
                tools=TOOL_DEFINITIONS,
            )
            message = completion.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                answer = (message.content or "").strip() or EMPTY_ANSWER
                return AgentResponse(
                    answer=answer,
                    stopped_reason="completed",
                    steps=steps,
                    pending_mutation=None,
                )

            messages.append(_assistant_tool_message(message))
            pending = self._apply_tool_calls(tool_calls, messages, steps)
            if pending is not None:
                return AgentResponse(
                    answer=PENDING_ANSWER,
                    stopped_reason="needs_approval",
                    steps=steps,
                    pending_mutation=pending,
                )

        return AgentResponse(
            answer=MAX_STEPS_ANSWER.format(n=self._max_steps),
            stopped_reason="max_steps",
            steps=steps,
            pending_mutation=None,
        )

    def _apply_tool_calls(
        self,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
        steps: list[AgentStep],
    ) -> dict[str, Any] | None:
        for call in tool_calls:
            name = call.function.name
            try:
                args = _parse_arguments(call.function.arguments)
            except ToolError as exc:
                self._record_error(call.id, name, {}, str(exc), messages, steps)
                continue

            if name in WRITE_TOOLS:
                try:
                    parse_args(ReceiptIdArgs, args)
                except ToolError as exc:
                    self._record_error(call.id, name, args, str(exc), messages, steps)
                    continue
                steps.append(
                    AgentStep(
                        tool=name,
                        args=args,
                        result=None,
                        status="needs_approval",
                    )
                )
                return {"tool": name, "args": args, "tool_call_id": call.id}

            try:
                payload = self._tools.dispatch(name, args)
                status = "ok"
            except ToolError as exc:
                payload = {"error": str(exc)}
                status = "error"
            steps.append(AgentStep(tool=name, args=args, result=payload, status=status))
            messages.append(_tool_message(call.id, payload))
        return None

    def _record_error(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        error: str,
        messages: list[dict[str, Any]],
        steps: list[AgentStep],
    ) -> None:
        payload = {"error": error}
        steps.append(AgentStep(tool=name, args=args, result=payload, status="error"))
        messages.append(_tool_message(call_id, payload))
        log.warning("agent_tool_error tool=%s error=%s", name, error[:200])
