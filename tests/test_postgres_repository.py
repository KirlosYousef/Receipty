from __future__ import annotations

import os

import pytest

from app.domain.schemas import Outcome, ReceiptExtract
from app.repository.receipts import PostgresReceiptRepository

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture
def repo() -> PostgresReceiptRepository:
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set; Postgres tests run only in Compose")
    repository = PostgresReceiptRepository(DATABASE_URL)
    repository.init_db()
    yield repository
    repository.close()


def test_save_and_list_roundtrip(repo: PostgresReceiptRepository) -> None:
    extract = ReceiptExtract(
        is_receipt=True,
        merchant="Postgres Cafe",
        total="12.50",
        currency="USD",
        date="2026-09-16",
        tax="1.25",
        outcome=Outcome.success,
    )

    repo.save(extract)
    rows = repo.list_all()

    assert rows, "list_all should return the saved receipt"
    assert rows[0]["merchant"] == "Postgres Cafe"
    assert rows[0]["total"] == "12.50"
    assert rows[0]["outcome"] == "success"


def test_non_receipt_roundtrip(repo: PostgresReceiptRepository) -> None:
    extract = ReceiptExtract(
        is_receipt=False,
        outcome=Outcome.not_receipt,
    )

    repo.save(extract)
    rows = repo.list_all()

    assert rows[0]["is_receipt"] is False
    assert rows[0]["merchant"] is None
    assert rows[0]["total"] is None
