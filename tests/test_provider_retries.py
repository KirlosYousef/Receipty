from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from collections.abc import Callable

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.llm.provider import OpenRouterProvider
from app.core.exceptions import ProviderDeadlineExceeded, CreditsExhausted, DailyLimitReached


class AlwaysConnectionError:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
            )
        )


def test_connection_error_uses_exact_attempt_limit_without_real_sleep():
    completions = AlwaysConnectionError()
    sleep_delays: list[float] = []

    settings = Settings(
        openrouter_api_key="test-key",
        max_attempts=3,
        total_deadline_seconds=30,
    )

    provider = OpenRouterProvider(
        settings,
        sleep_fn=sleep_delays.append,
        jitter_fn=lambda maximum: maximum / 2,
        clock=lambda: 0.0,
    )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with pytest.raises(ProviderError):
        provider.complete(
            [{"role": "user", "content": "receipt"}]
        )

    assert completions.calls == 3
    assert sleep_delays == [0.5, 1.0]

class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def test_total_deadline_caps_retries_and_sleep():
    completions = AlwaysConnectionError()
    fake_time = FakeTime()

    settings = Settings(
        openrouter_api_key="test-key",
        max_attempts=5,
        request_timeout_seconds=60,
        total_deadline_seconds=1.5,
        retry_base_delay_seconds=1,
    )

    provider = OpenRouterProvider(
        settings,
        sleep_fn=fake_time.sleep,
        jitter_fn=lambda maximum: maximum,
        clock=fake_time.clock,
    )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with pytest.raises(ProviderDeadlineExceeded):
        provider.complete(
            [{"role": "user", "content": "receipt"}]
        )

    assert completions.calls == 2
    assert fake_time.sleeps == [1.0, 0.5]
    

class AlwaysRaises:
    def __init__(self, error_factory: Callable[[], Exception]):
        self.error_factory = error_factory
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self.error_factory()


def status_error(
    status_code: int,
    message: str,
    error_type=APIStatusError,
):
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    response = httpx.Response(
        status_code,
        request=request,
    )

    return error_type(
        message,
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    ("error_factory", "expected_error"),
    [
        (
            lambda: status_error(402, "Payment required"),
            CreditsExhausted,
        ),
        (
            lambda: status_error(
                429,
                "Daily limit reached: limit_rpd",
                RateLimitError,
            ),
            DailyLimitReached,
        ),
        (
            lambda: status_error(400, "Bad request"),
            ProviderError,
        ),
    ],
)
def test_non_retryable_errors_fail_after_one_attempt(
    error_factory,
    expected_error,
):
    completions = AlwaysRaises(error_factory)
    sleep_delays: list[float] = []

    provider = OpenRouterProvider(
        Settings(
            openrouter_api_key="test-key",
            max_attempts=3,
        ),
        sleep_fn=sleep_delays.append,
        jitter_fn=lambda maximum: maximum,
        clock=lambda: 0.0,
    )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with pytest.raises(expected_error):
        provider.complete(
            [{"role": "user", "content": "receipt"}]
        )

    assert completions.calls == 1
    assert sleep_delays == []


def test_server_errors_retry_until_attempt_limit():
    completions = AlwaysRaises(
        lambda: status_error(500, "Server error")
    )
    sleep_delays: list[float] = []

    provider = OpenRouterProvider(
        Settings(
            openrouter_api_key="test-key",
            max_attempts=3,
        ),
        sleep_fn=sleep_delays.append,
        jitter_fn=lambda maximum: maximum,
        clock=lambda: 0.0,
    )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    with pytest.raises(ProviderError):
        provider.complete(
            [{"role": "user", "content": "receipt"}]
        )

    assert completions.calls == 3
    assert sleep_delays == [1.0, 2.0]


def test_sdk_retries_are_disabled(monkeypatch):
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.llm.provider.OpenAI",
        FakeOpenAI,
    )

    OpenRouterProvider(
        Settings(openrouter_api_key="test-key")
    )

    assert captured["max_retries"] == 0
