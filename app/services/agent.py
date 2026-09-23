from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator, Iterator
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.exceptions import ProviderError
from app.domain.schemas import AgentResponse, AgentStep
from app.llm.prompts import AGENT_PROMPT
from app.llm.provider import LLMProvider
from app.observability.tracing import bind_request_id
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
WRITE_APPLIED_ANSWER = "Write applied."
WRITE_REJECTED_ANSWER = "Write rejected."
FALLBACK_TIMEOUT = "Stopped because this agent run ran out of time."
FALLBACK_PROVIDER = "Stopped because the model provider failed."


class AgentNotPaused(ValueError):
    """Resume was called for a thread that is not waiting on a write."""


class AgentGraphState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    pending_calls: list[dict[str, Any]]
    pending_mutation: dict[str, Any] | None
    answer: str
    stopped_reason: str
    rounds: int
    request_id: str | None
    started_at: float


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


def _serialize_tool_calls(message: Any) -> list[dict[str, Any]]:
    serialized = []
    for call in message.tool_calls:
        arguments = call.function.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        serialized.append(
            {
                "id": call.id,
                "name": call.function.name,
                "arguments": arguments,
            }
        )
    return serialized


def _tool_message(call_id: str, payload: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, default=str),
    }


def _approved(decision: Any) -> bool:
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return bool(decision)


class AgentService:
    def __init__(
        self,
        provider: LLMProvider,
        tools: AgentTools,
        *,
        max_steps: int = 8,
        deadline_seconds: float = 60.0,
        prompt: str = AGENT_PROMPT,
        checkpointer: Any | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._provider = provider
        self._tools = tools
        self._max_steps = max_steps
        self._deadline_seconds = deadline_seconds
        self._prompt = prompt
        self._clock = clock or time.monotonic
        self._graph = self._build_graph(checkpointer or InMemorySaver())

    def run(
        self,
        question: str,
        *,
        request_id: str | None = None,
        thread_id: str | None = None,
    ) -> AgentResponse:
        thread_id = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        with bind_request_id(request_id):
            self._graph.invoke(self._initial_state(question, request_id), config)
        return self._response(thread_id, config)

    def stream(
        self,
        question: str,
        *,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield each new tool step, then one final event for the same result as run()."""
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        seen = 0
        for chunk in self._graph.stream(
            self._initial_state(question, request_id),
            config,
            stream_mode="updates",
        ):
            seen = yield from self._emit_new_steps(config, seen)
            if "__interrupt__" in chunk:
                yield self._done_event(thread_id, config)
                return
        yield self._done_event(thread_id, config)

    def _initial_state(self, question: str, request_id: str | None) -> AgentGraphState:
        return {
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": question},
            ],
            "steps": [],
            "pending_calls": [],
            "pending_mutation": None,
            "answer": "",
            "stopped_reason": "",
            "rounds": 0,
            "request_id": request_id,
            "started_at": self._clock(),
        }

    def _emit_new_steps(
        self, config: dict[str, Any], seen: int
    ) -> Generator[dict[str, Any], None, int]:
        snapshot = self._graph.get_state(config)
        steps = (snapshot.values or {}).get("steps") or []
        for step in steps[seen:]:
            yield {"event": "step", "data": step}
        return len(steps)

    def _done_event(self, thread_id: str, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": "done",
            "data": self._response(thread_id, config).model_dump(mode="json"),
        }

    def resume(
        self,
        thread_id: str,
        *,
        approved: bool,
        request_id: str | None = None,
    ) -> AgentResponse:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)
        if not snapshot.values:
            raise LookupError(f"agent thread {thread_id} not found")
        if snapshot.next != ("apply_write",):
            raise AgentNotPaused("agent thread is not waiting for approval")
        self._graph.update_state(config, {"started_at": self._clock()})
        with bind_request_id(request_id):
            self._graph.invoke(Command(resume={"approved": approved}), config)
        return self._response(thread_id, config)

    def _response(self, thread_id: str, config: dict[str, Any]) -> AgentResponse:
        snapshot = self._graph.get_state(config)
        values = snapshot.values or {}
        steps = [AgentStep.model_validate(step) for step in values.get("steps") or []]
        if snapshot.next:
            return AgentResponse(
                answer=values.get("answer") or PENDING_ANSWER,
                stopped_reason="needs_approval",
                steps=steps,
                pending_mutation=values.get("pending_mutation"),
                thread_id=thread_id,
            )
        stopped = values.get("stopped_reason")
        reason: Literal["completed", "max_steps", "needs_approval", "fallback"] = (
            "completed"
        )
        if stopped in ("completed", "max_steps", "needs_approval", "fallback"):
            reason = stopped
        return AgentResponse(
            answer=values.get("answer") or EMPTY_ANSWER,
            stopped_reason=reason,
            steps=steps,
            pending_mutation=None,
            thread_id=thread_id,
        )

    def _build_graph(self, checkpointer: Any) -> Any:
        builder = StateGraph(AgentGraphState)
        builder.add_node("call_model", self._call_model)
        builder.add_node("apply_tools", self._apply_tools)
        builder.add_node("apply_write", self._apply_write)
        builder.add_node("max_steps", self._max_steps_node)
        builder.add_node("finish_write", self._finish_write)
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {"apply_tools": "apply_tools", END: END},
        )
        builder.add_conditional_edges(
            "apply_tools",
            self._route_after_tools,
            {
                "apply_write": "apply_write",
                "call_model": "call_model",
                "max_steps": "max_steps",
            },
        )
        builder.add_conditional_edges(
            "apply_write",
            self._route_after_write,
            {"call_model": "call_model", "finish_write": "finish_write"},
        )
        builder.add_edge("max_steps", END)
        builder.add_edge("finish_write", END)
        return builder.compile(checkpointer=checkpointer)

    def _call_model(self, state: AgentGraphState) -> AgentGraphState:
        if self._out_of_time(state):
            return self._fallback(FALLBACK_TIMEOUT)
        messages = list(state.get("messages") or [])
        try:
            completion = self._provider.complete(
                messages,
                request_id=state.get("request_id"),
                tools=TOOL_DEFINITIONS,
            )
        except ProviderError:
            log.warning("agent_provider_failed request_id=%s", state.get("request_id"))
            return self._fallback(FALLBACK_PROVIDER)
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        rounds = int(state.get("rounds") or 0) + 1
        if not tool_calls:
            answer = (message.content or "").strip() or EMPTY_ANSWER
            return {
                "rounds": rounds,
                "answer": answer,
                "stopped_reason": "completed",
                "pending_calls": [],
                "pending_mutation": None,
            }
        messages.append(_assistant_tool_message(message))
        return {
            "rounds": rounds,
            "messages": messages,
            "pending_calls": _serialize_tool_calls(message),
            "stopped_reason": "",
            "pending_mutation": None,
        }

    def _apply_tools(self, state: AgentGraphState) -> AgentGraphState:
        messages = list(state.get("messages") or [])
        steps = list(state.get("steps") or [])
        pending_mutation: dict[str, Any] | None = None

        for call in state.get("pending_calls") or []:
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or "")
            try:
                args = _parse_arguments(call.get("arguments"))
            except ToolError as exc:
                self._append_error(call_id, name, {}, str(exc), messages, steps)
                continue

            if name in WRITE_TOOLS:
                try:
                    parse_args(ReceiptIdArgs, args)
                except ToolError as exc:
                    self._append_error(call_id, name, args, str(exc), messages, steps)
                    continue
                steps.append(
                    {
                        "tool": name,
                        "args": args,
                        "result": None,
                        "status": "needs_approval",
                    }
                )
                pending_mutation = {
                    "tool": name,
                    "args": args,
                    "tool_call_id": call_id,
                }
                break

            try:
                payload = self._dispatch(state, name, args)
                status: Literal["ok", "error"] = "ok"
            except ToolError as exc:
                payload = {"error": str(exc)}
                status = "error"
            steps.append(
                {"tool": name, "args": args, "result": payload, "status": status}
            )
            messages.append(_tool_message(call_id, payload))

        if pending_mutation is not None:
            return {
                "messages": messages,
                "steps": steps,
                "pending_calls": [],
                "pending_mutation": pending_mutation,
                "answer": PENDING_ANSWER,
                "stopped_reason": "needs_approval",
            }
        return {
            "messages": messages,
            "steps": steps,
            "pending_calls": [],
            "pending_mutation": None,
        }

    def _dispatch(self, state: AgentGraphState, name: str, args: dict[str, Any]) -> Any:
        with bind_request_id(state.get("request_id")):
            return self._tools.dispatch(name, args)

    def _apply_write(self, state: AgentGraphState) -> AgentGraphState:
        pending = state.get("pending_mutation")
        if not pending:
            return {"pending_mutation": None}
        # interrupt() must run before any ledger write. On resume this node
        # restarts from the top; a write above this line would run twice.
        decision = interrupt(pending)
        steps = list(state.get("steps") or [])
        messages = list(state.get("messages") or [])
        if _approved(decision):
            try:
                payload = self._dispatch(state, pending["tool"], pending["args"])
                status = "ok"
            except ToolError as exc:
                payload = {"error": str(exc)}
                status = "error"
                log.warning(
                    "agent_write_failed tool=%s error=%s",
                    pending["tool"],
                    str(exc)[:200],
                )
        else:
            payload = {"error": "write rejected"}
            status = "error"
        if steps and steps[-1].get("status") == "needs_approval":
            steps[-1] = {
                **steps[-1],
                "result": payload,
                "status": status,
            }
        else:
            steps.append(
                {
                    "tool": pending["tool"],
                    "args": pending["args"],
                    "result": payload,
                    "status": status,
                }
            )
        messages.append(_tool_message(str(pending["tool_call_id"]), payload))
        return {
            "messages": messages,
            "steps": steps,
            "pending_mutation": None,
            "stopped_reason": "",
        }

    def _max_steps_node(self, state: AgentGraphState) -> AgentGraphState:
        del state
        return {
            "answer": MAX_STEPS_ANSWER.format(n=self._max_steps),
            "stopped_reason": "max_steps",
            "pending_mutation": None,
        }

    def _finish_write(self, state: AgentGraphState) -> AgentGraphState:
        steps = state.get("steps") or []
        last_status = steps[-1].get("status") if steps else "error"
        answer = WRITE_APPLIED_ANSWER if last_status == "ok" else WRITE_REJECTED_ANSWER
        return {
            "answer": answer,
            "stopped_reason": "completed",
            "pending_mutation": None,
        }

    def _out_of_time(self, state: AgentGraphState) -> bool:
        started = state.get("started_at")
        if started is None:
            return False
        return self._clock() - float(started) >= self._deadline_seconds

    def _fallback(self, answer: str) -> AgentGraphState:
        return {
            "answer": answer,
            "stopped_reason": "fallback",
            "pending_calls": [],
            "pending_mutation": None,
        }

    def _route_after_model(
        self, state: AgentGraphState
    ) -> Literal["apply_tools"] | Any:
        if state.get("stopped_reason") in ("completed", "fallback"):
            return END
        return "apply_tools"

    def _route_after_tools(
        self, state: AgentGraphState
    ) -> Literal["apply_write", "call_model", "max_steps"]:
        if state.get("pending_mutation"):
            return "apply_write"
        if int(state.get("rounds") or 0) >= self._max_steps:
            return "max_steps"
        return "call_model"

    def _route_after_write(
        self, state: AgentGraphState
    ) -> Literal["call_model", "finish_write"]:
        if int(state.get("rounds") or 0) >= self._max_steps:
            return "finish_write"
        return "call_model"

    def _append_error(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        error: str,
        messages: list[dict[str, Any]],
        steps: list[dict[str, Any]],
    ) -> None:
        payload = {"error": error}
        steps.append({"tool": name, "args": args, "result": payload, "status": "error"})
        messages.append(_tool_message(call_id, payload))
        log.warning("agent_tool_error tool=%s error=%s", name, error[:200])
