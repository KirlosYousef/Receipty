from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderError
from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider, OpenRouterEmbeddings
from app.repository.receipts import EMBEDDING_DIMENSIONS, SqliteReceiptRepository
from app.services.indexing import IndexingService, receipt_document_text


def test_hash_embedding_has_fixed_dimension():
    vector = HashEmbeddingProvider().embed("Receipt 1. Merchant: Test Cafe.")
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6


def test_receipt_document_text_includes_core_fields():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Test Cafe",
        total=Decimal("12.50"),
        currency="USD",
        date="2024-01-15",
        outcome=Outcome.success,
    )
    text = receipt_document_text(7, row)
    assert "Receipt 7" in text
    assert "Test Cafe" in text
    assert "12.50" in text
    assert "USD" in text
    assert "2024-01-15" in text


def test_seed_and_receipt_index_write_documents(tmp_path):
    repo = SqliteReceiptRepository(tmp_path / "receipts.db")
    repo.init_db()
    indexer = IndexingService(repo, HashEmbeddingProvider())

    indexer.seed_static_documents()
    indexer.seed_static_documents()

    extract = ReceiptExtract(
        is_receipt=True,
        merchant="Test Cafe",
        total=Decimal("12.50"),
        currency="USD",
        date="2024-01-15",
        outcome=Outcome.success,
    )
    receipt_id = repo.save(extract)
    indexer.index_receipt(receipt_id, extract)

    documents = repo.list_documents()
    kinds = {row["kind"] for row in documents}
    source_ids = {row["source_id"] for row in documents}

    assert "merchant_alias" in kinds
    assert "policy_note" in kinds
    assert "receipt" in kinds
    assert f"receipt:{receipt_id}" in source_ids
    assert len([row for row in documents if row["kind"] == "merchant_alias"]) == 5
    assert len([row for row in documents if row["kind"] == "policy_note"]) == 3


def test_openrouter_embeddings_require_api_key():
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY is not set"):
        OpenRouterEmbeddings(Settings(openrouter_api_key=""))


def test_openrouter_embed_rejects_wrong_dimension():
    provider = OpenRouterEmbeddings(Settings(openrouter_api_key="test-key"))
    fake_response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    provider._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: fake_response)
    )
    with pytest.raises(ValueError, match="dimension"):
        provider.embed("receipt")


def test_non_receipt_is_not_indexed(tmp_path):
    repo = SqliteReceiptRepository(tmp_path / "receipts.db")
    repo.init_db()
    indexer = IndexingService(repo, HashEmbeddingProvider())
    extract = ReceiptExtract(is_receipt=False, outcome=Outcome.not_receipt)
    receipt_id = repo.save(extract)
    indexer.index_receipt(receipt_id, extract)
    assert repo.list_documents() == []
