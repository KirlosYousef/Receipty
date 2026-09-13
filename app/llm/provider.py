from __future__ import annotations

import logging
import random
import time
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.core.config import Settings
from app.core.exceptions import CreditsExhausted, DailyLimitReached, ProviderDeadlineExceeded, ProviderError
from app.domain.schemas import ReceiptLLMOutput

from collections.abc import Callable

log = logging.getLogger(__name__)

def _full_jitter(maximum: float) -> float:
    return random.uniform(0, maximum)

class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> Any: ...

    def close(self) -> None: ...

class OpenRouterProvider:
    def __init__(
            self,
            settings: Settings,
            *,
            sleep_fn: Callable[[float], None] | None = None,
            jitter_fn: Callable[[float], float] | None = None,
            clock: Callable[[], float] | None = None,
        ):
            if not settings.openrouter_api_key:
                raise ProviderError("OPENROUTER_API_KEY is not set")

            self._settings = settings
            self._sleep = sleep_fn or time.sleep
            self._jitter = jitter_fn or _full_jitter
            self._clock = clock or time.monotonic

            self._client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
                max_retries=0,
            )
    
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> Any:
        delay = 1.0
        started_at = self._clock()
        last: BaseException | None = None

        for attempt in range(1, self._settings.max_attempts + 1):
            elapsed = self._clock() - started_at
            remaining = self._settings.total_deadline_seconds - elapsed

            if remaining <= 0:
                raise ProviderDeadlineExceeded("Provider deadline exhausted")

            attempt_timeout = min(
                self._settings.request_timeout_seconds,
                remaining,
            )

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
                timeout=attempt_timeout,

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

            if attempt == self._settings.max_attempts:
                log.error(
                    "provider_failed request_id=%s attempt=%s max_attempts=%s error=%s",
                    request_id,
                    attempt,
                    self._settings.max_attempts,
                    type(last).__name__,
                )
                raise ProviderError(str(last)) from last

            delay_cap = (
                self._settings.retry_base_delay_seconds
                * (2 ** (attempt - 1))
            )
            delay = self._jitter(delay_cap)

            elapsed = self._clock() - started_at
            remaining = self._settings.total_deadline_seconds - elapsed

            if remaining <= 0:
                raise ProviderDeadlineExceeded("Provider deadline exhausted")

            log.warning(
                "provider_retry request_id=%s attempt=%s max_attempts=%s delay_seconds=%.3f error=%s",
                request_id,
                attempt,
                self._settings.max_attempts,
                delay,
                type(last).__name__,
            )
            self._sleep(min(delay, remaining))


        if last is None:
            raise ProviderError("complete() failed with no exception")
        raise ProviderError(str(last)) from last

    def close(self) -> None:
        self._client.close()