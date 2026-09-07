from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_extraction_service
from app.main import create_app
from app.repository.receipts import ReceiptRepository
from app.services.extraction import ExtractionService


class FakeProvider:
    def __init__(self, content: str):
        self.content = content

    def complete(self, messages):
        return SimpleNamespace(
            model="fake",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0, model_extra={}),
        )


@pytest.fixture
def client(tmp_path: Path):
    app = create_app()
    repo = ReceiptRepository(tmp_path / "test.db")
    repo.init_db()

    def fake_service():
        provider = FakeProvider(
            '{"is_receipt": true, "merchant": "Test Cafe", "total": "12.50", '
            '"currency": "USD", "date": "2024-01-15", "tax": "1.25", "needs_review": false}'
        )
        return ExtractionService(provider)

    app.dependency_overrides[get_extraction_service] = fake_service
    with TestClient(app) as c:
        app.state.repo = repo
        yield c
    app.dependency_overrides.clear()


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


def test_extraction_service_bad_json():
    provider = FakeProvider("not-json")
    svc = ExtractionService(provider)
    row = svc.extract_from_text("hello")
    assert row.is_receipt is False
    assert row.needs_review is True
