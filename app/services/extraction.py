from __future__ import annotations

import base64
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from app.domain.schemas import Outcome, ReceiptExtract, ReceiptLLMOutput
from app.llm.prompts import EXTRACTION_PROMPT
from app.llm.provider import LLMProvider
from app.services.postprocess import apply_postprocess

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_json_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip())


class UsageLogger(Protocol):
    def log(self, completion: Any, kind: str) -> None: ...


class ExtractionService:
    def __init__(self, provider: LLMProvider, usage: UsageLogger | None = None):
        self._provider = provider
        self._usage = usage

    def extract_from_text(
        self,
        text: str,
        *,
        request_id: str | None = None,
    ) -> ReceiptExtract:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ]
        return self._run(
            messages,
            currency_hint=text,
            kind="text",
            request_id=request_id,
        )

    def extract_from_image(
        self,
        image_bytes: bytes,
        mime: str,
        *,
        request_id: str | None = None,
    ) -> ReceiptExtract:
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url}}],
            },
        ]
        return self._run(
            messages,
            currency_hint=mime,
            kind="image",
            request_id=request_id,
        )

    def _run(
        self,
        messages: list[dict[str, Any]],
        *,
        currency_hint: str,
        kind: str,
        request_id: str | None = None,
    ) -> ReceiptExtract:
        completion = self._provider.complete(
            messages,
            request_id=request_id,
        )
        content = completion.choices[0].message.content
        if content is None:
            return ReceiptExtract(is_receipt=False, outcome=Outcome.extraction_failed)
        try:
            llm_output = ReceiptLLMOutput.model_validate_json(
                _strip_json_fences(content)
            )

            row = ReceiptExtract.model_validate(llm_output.model_dump())
        except ValidationError:
            # Refuse to invent fields when the model returns garbage JSON.
            log.warning("bad model json: %s", content[:300])
            row = ReceiptExtract(is_receipt=False, outcome=Outcome.extraction_failed)

        if self._usage is not None:
            self._usage.log(completion, kind)

        return apply_postprocess(row, currency_hint)
