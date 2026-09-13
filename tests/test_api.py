from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.routes import _map_provider_error
from app.core.config import Settings
from app.core.exceptions import ProviderDeadlineExceeded
from app.main import create_app


class FakeProvider:
    def __init__(self, content: str):
        self.content = content
        self.request_ids: list[str | None] = []

    def complete(self, messages, *, request_id=None):
        self.request_ids.append(request_id)

        return SimpleNamespace(
            model="fake",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                cost=0,
                model_extra={},
            ),
        )

    def close(self) -> None:
        pass


@pytest.fixture
def client(tmp_path: Path):
    fake_provider = FakeProvider(
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
    )

    settings = Settings(
        openrouter_api_key="test-key",
        db_path=tmp_path / "test.db",
        cost_log_path=tmp_path / "cost.jsonl",
    )

    app = create_app(
        settings=settings,
        provider_factory=lambda _: fake_provider,
    )

    app.state.test_provider = fake_provider

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "Receipty" in r.text
    assert "/assets/styles.css" in r.text


def test_ingest_text(client: TestClient):
    r = client.post("/v1/ingest", json={"text": "Test Cafe TOTAL 12.50 USD"})
    assert r.status_code == 200
    body = r.json()["extract"]
    assert body["is_receipt"] is True
    assert body["merchant"] == "Test Cafe"
    assert body["total"] == "12.50"
    assert body["date"] == "2024-01-15"
    assert body["tax"] == "1.25"


def test_ingest_image(client: TestClient):
    r = client.post(
        "/v1/ingest/image",
        files={"file": ("r.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert r.status_code == 200
    assert r.json()["extract"]["merchant"] == "Test Cafe"


def test_ingest_image_bad_mime(client: TestClient):
    r = client.post(
        "/v1/ingest/image",
        files={"file": ("r.gif", b"fake", "image/gif")},
    )
    assert r.status_code == 400


def test_list_receipts(client: TestClient):
    client.post("/v1/ingest", json={"text": "Test Cafe TOTAL 12.50 USD"})
    r = client.get("/v1/receipts")
    assert r.status_code == 200
    assert len(r.json()) >= 1
    assert r.json()[0]["tax"] == "1.25"


def test_response_contains_generated_request_id(client: TestClient):
    response = client.get("/health")

    request_id = response.headers["X-Request-ID"]

    # Raises ValueError if it is not a valid UUID.
    UUID(request_id)


def test_incoming_request_id_is_preserved(client: TestClient):
    response = client.get(
        "/health",
        headers={"X-Request-ID": "client-request-123"},
    )

    assert response.headers["X-Request-ID"] == "client-request-123"


def test_ingest_propagates_request_id(client: TestClient):
    response = client.post(
        "/v1/ingest",
        json={"text": "Test Cafe Total 12.50 USD"},
        headers={"X-Request-ID": "ingest-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "ingest-request-123"
    assert client.app.state.test_provider.request_ids == ["ingest-request-123"]


def test_provider_deadline_maps_to_gateway_timeout():
    error = _map_provider_error(ProviderDeadlineExceeded("Provider deadline exhausted"))

    assert error.status_code == 504
    assert error.detail == "Provider deadline exhausted"
