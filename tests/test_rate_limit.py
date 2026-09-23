from pathlib import Path

from fastapi.testclient import TestClient

from app.api.rate_limit import RateLimiter
from app.core.config import Settings
from app.llm.embeddings import HashEmbeddingProvider
from app.main import create_app
from tests.test_api import FakeProvider


def test_window_drops_hits_that_are_older_than_the_limit():
    times = iter([0.0, 0.0, 0.0, 61.0])
    limiter = RateLimiter(1, 60, clock=lambda: next(times))
    assert limiter.allow("10.0.0.1") is True
    assert limiter.allow("10.0.0.1") is False
    assert limiter.allow("10.0.0.8") is True
    assert limiter.allow("10.0.0.1") is True


def test_model_routes_return_429_and_health_stays_open(tmp_path: Path):
    settings = Settings(
        openrouter_api_key="test-key",
        db_path=tmp_path / "test.db",
        cost_log_path=tmp_path / "cost.jsonl",
        rate_limit_requests=1,
        rate_limit_window_seconds=60,
    )
    app = create_app(
        settings=settings,
        provider_factory=lambda _: FakeProvider(
            """
            {
                "is_receipt": true,
                "merchant": "Test Cafe",
                "total": "12.50",
                "currency": "USD",
                "date": "2024-01-15",
                "tax": "1.25"
            }
            """
        ),
        embedding_factory=lambda _: HashEmbeddingProvider(),
    )
    with TestClient(app) as client:
        first = client.post("/v1/ingest", json={"text": "not a receipt"})
        second = client.post("/v1/ingest", json={"text": "not a receipt"})
        health = client.get("/health")
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Try again later."
    assert second.headers["retry-after"] == "60"
    assert health.status_code == 200
