from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from app.domain.schemas import AnswerResponse
from app.llm.prompts import ANSWER_PROMPT
from app.llm.provider import LLMProvider
from app.services.retrieval import RetrievalService

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


def build_context(documents: list[dict]) -> str:
    """Render retrieved documents into a single context block for the LLM."""
    if not documents:
        return "Context is empty."
    lines = []
    for doc in documents:
        source_id = doc["source_id"]
        content = doc["content"]
        lines.append(f"[source_id: {source_id}] {content}")
    return "\n".join(lines)


NOT_FOUND_ANSWER = AnswerResponse(
    answer="I could not find a relevant receipt for that question.",
    citations=[],
    found=False,
)


class AnsweringService:
    def __init__(
        self,
        provider: LLMProvider,
        retrieval: RetrievalService,
        *,
        prompt: str = ANSWER_PROMPT,
    ):
        self._provider = provider
        self._retrieval = retrieval
        self._prompt = prompt

    def answer(self, question: str, *, request_id: str | None = None) -> AnswerResponse:
        documents = self._retrieval.search(question, strategy="hybrid", limit=5)
        if not documents:
            return NOT_FOUND_ANSWER

        context = build_context(documents)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

        try:
            completion = self._provider.complete(
                messages,
                request_id=request_id,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": AnswerResponse.model_json_schema(),
                    },
                },
            )
        except Exception:
            log.exception("answer_provider_failed request_id=%s", request_id)
            return NOT_FOUND_ANSWER

        content = completion.choices[0].message.content
        if content is None:
            return NOT_FOUND_ANSWER

        try:
            return AnswerResponse.model_validate_json(_strip_json_fences(content))
        except ValidationError:
            log.warning("bad answer json: %s", content[:300])
            return NOT_FOUND_ANSWER
