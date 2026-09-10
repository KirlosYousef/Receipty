from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.core.config import Settings
from app.core.exceptions import CreditsExhausted, DailyLimitReached, ProviderError
from app.domain.schemas import ReceiptLLMOutput

log = logging.getLogger(__name__)


class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> Any: ...
    def close(self) -> None: ...

class OpenRouterProvider:
    def __init__(self, settings: Settings):
        if not settings.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        delay = 1.0
        last: BaseException | None = None
        for attempt in range(1, self._settings.max_retries + 1):
            try:
                return self._client.chat.completions.create(
                    model=self._settings.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "receipt_extraction",
                            "strict": True,
                            "schema": ReceiptLLMOutput.model_json_schema(),
                        },
                    },
                    extra_body={
                        "provider": {
                            "require_parameters": True,
                        }
                    },
                    timeout=self._settings.request_timeout_seconds,
                )
            except RateLimitError as e:
                msg = str(e)
                if "Daily limit" in msg or "limit_rpd" in msg:
                    raise DailyLimitReached(
                        "Daily limit reached for this free model. "
                        "Credits do not lift this cap. Try tomorrow or another model."
                    ) from e
                last = e
            except APIStatusError as e:
                if e.status_code == 402:
                    raise CreditsExhausted("OpenRouter credits exhausted") from e
                if e.status_code is None or e.status_code < 500:
                    raise ProviderError(str(e)) from e
                last = e
            except APIConnectionError as e:
                last = e

            if attempt == self._settings.max_retries:
                break
            log.warning(
                "OpenRouter retry %s/%s in %.0fs (%s)",
                attempt,
                self._settings.max_retries,
                delay,
                last,
            )
            time.sleep(delay)
            delay *= 2

        if last is None:
            raise ProviderError("complete() failed with no exception")
        raise ProviderError(str(last)) from last

    def close(self) -> None:
        self._client.close()