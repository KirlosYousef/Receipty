from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.main import create_app


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def complete(self, messages, *, request_id=None):
        self.calls += 1

        return SimpleNamespace(
            model="fake",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"is_receipt": true,'
                            '"merchant": "Test Cafe",'
                            '"total": "12.50",'
                            '"currency": "USD",'
                            '"date": null,'
                            '"tax": null}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                cost=0,
                model_extra={},
            ),
        )

    def close(self):
        self.closed = True


def test_provider_is_reused_and_closed(tmp_path: Path):
    provider = FakeProvider()
    factory_calls = 0

    def provider_factory(settings):
        nonlocal factory_calls
        factory_calls += 1
        return provider

    settings = Settings(
        openrouter_api_key="test-key",
        db_path=tmp_path / "receipts.db",
        cost_log_path=tmp_path / "cost.jsonl",
    )

    app = create_app(
        settings=settings,
        provider_factory=provider_factory,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/ingest",
            json={"text": "Test Cafe Total 12.50 USD"},
        )
        second = client.post(
            "/v1/ingest",
            json={"text": "Test Cafe Total 12.50 USD"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert factory_calls == 1
        assert provider.calls == 2
        assert provider.closed is False

    assert provider.closed is True


def test_missing_api_key_fails_during_startup(tmp_path: Path):
    settings = Settings(
        openrouter_api_key="",
        db_path=tmp_path / "receipts.db",
        cost_log_path=tmp_path / "cost.jsonl",
    )

    app = create_app(settings=settings)

    with pytest.raises(
        ProviderError,
        match="OPENROUTER_API_KEY is not set",
    ):
        with TestClient(app):
            pass
