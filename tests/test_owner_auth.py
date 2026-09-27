import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.embeddings import HashEmbeddingProvider
from app.main import create_app
from tests.test_api import FakeProvider

PASSWORD = "test-only-long-owner-password"


def _authorization(password: str = PASSWORD) -> dict[str, str]:
    encoded = base64.b64encode(f"owner:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@pytest.fixture
def protected_client(tmp_path: Path):
    provider = FakeProvider(
        '{"is_receipt": true, "merchant": "Test Cafe", "total": "12.50", '
        '"currency": "USD", "date": "2024-01-15", "tax": "1.25"}'
    )
    settings = Settings(
        _env_file=None,
        auth_required=True,
        auth_password=PASSWORD,
        openrouter_api_key="test-key",
        db_path=tmp_path / "receipts.db",
        cost_log_path=tmp_path / "cost.jsonl",
        trace_log_path=tmp_path / "traces.jsonl",
    )
    app = create_app(
        settings=settings,
        provider_factory=lambda _: provider,
        embedding_factory=lambda _: HashEmbeddingProvider(),
    )
    with TestClient(app) as client:
        yield client, provider


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("GET", "/", {}),
        ("GET", "/assets/app.js", {}),
        ("GET", "/docs", {}),
        ("GET", "/openapi.json", {}),
        ("GET", "/v1/receipts", {}),
        ("GET", "/v1/usage", {}),
        ("GET", "/v1/search?q=cafe", {}),
        ("POST", "/v1/ingest", {"json": {"text": "Test Cafe TOTAL 12.50"}}),
        (
            "POST",
            "/v1/ingest/image",
            {"files": {"file": ("receipt.jpg", b"\xff\xd8\xffbytes", "image/jpeg")}},
        ),
        ("POST", "/v1/agent", {"json": {"question": "Count receipts"}}),
        ("POST", "/v1/agent/stream", {"json": {"question": "Count receipts"}}),
        (
            "POST",
            "/v1/agent/resume",
            {"json": {"thread_id": "unknown", "approved": True}},
        ),
    ],
)
def test_anonymous_access_is_rejected_before_work(
    protected_client, method: str, path: str, kwargs: dict
):
    client, provider = protected_client
    response = client.request(method, path, **kwargs)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Basic realm="Receipty"'
    assert response.headers["x-request-id"]
    assert provider.request_ids == []
    assert client.app.state.repo.list_all() == []


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer any",
        "Basic !!!",
        "Basic bm9jb2xvbg==",
        "Basic /zpwYXNz",
        _authorization("wrong")["Authorization"],
    ],
)
def test_invalid_credentials_are_rejected(protected_client, authorization: str):
    client, _ = protected_client
    response = client.get("/v1/receipts", headers={"Authorization": authorization})
    assert response.status_code == 401


def test_correct_credentials_allow_dashboard_and_ingest(protected_client):
    client, provider = protected_client
    assert client.get("/", headers=_authorization()).status_code == 200
    response = client.post(
        "/v1/ingest",
        json={"text": "Test Cafe TOTAL 12.50"},
        headers=_authorization(),
    )
    assert response.status_code == 200
    assert len(provider.request_ids) == 1
    assert len(client.get("/v1/receipts", headers=_authorization()).json()) == 1


def test_health_remains_public(protected_client):
    client, _ = protected_client
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("password", ["", "short"])
def test_required_auth_without_strong_password_fails_startup(
    tmp_path: Path, password: str
):
    settings = Settings(
        _env_file=None,
        auth_required=True,
        auth_password=password,
        db_path=tmp_path / "receipts.db",
    )
    app = create_app(settings=settings)
    with pytest.raises(RuntimeError, match="AUTH_PASSWORD must contain at least 16"):
        with TestClient(app):
            pass
