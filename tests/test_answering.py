from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.domain.schemas import AnswerResponse, Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.repository.receipts import SqliteReceiptRepository
from app.services.answering import NOT_FOUND_ANSWER, AnsweringService, build_context
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository


class FakeProvider:
    def __init__(self, content: str | None):
        self.content = content

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        request_id: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )

    def close(self) -> None:
        pass


def _seed(tmp_path: Path) -> SqliteReceiptRepository:
    repo = SqliteReceiptRepository(tmp_path / "receipts.db")
    repo.init_db()
    indexer = IndexingService(repo, HashEmbeddingProvider())
    indexer.seed_static_documents()
    extract = ReceiptExtract(
        is_receipt=True,
        merchant="Taco Bell",
        total="7.61",
        currency="USD",
        date="2016-09-01",
        outcome=Outcome.success,
    )
    receipt_id = repo.save(extract)
    indexer.index_receipt(receipt_id, extract)
    return repo


def _build_answering(tmp_path: Path, provider: FakeProvider) -> AnsweringService:
    repo = SqliteRetrievalRepository(tmp_path / "receipts.db")
    retrieval = RetrievalService(repo, HashEmbeddingProvider())
    return AnsweringService(provider, retrieval)


def test_build_context_tags_documents_with_source_id():
    docs = [
        {"source_id": "receipt:5", "content": "Receipt 5. Merchant: Taco Bell."},
        {"source_id": "policy:p1", "content": "Never invent a total."},
    ]
    context = build_context(docs)
    assert "[source_id: receipt:5]" in context
    assert "[source_id: policy:p1]" in context


def test_build_context_empty():
    assert build_context([]) == "Context is empty."


def test_answer_cites_retrieved_receipt(tmp_path: Path):
    _seed(tmp_path)
    llm_json = '{"answer": "Taco Bell total was 7.61 USD.", "citations": ["receipt:1"], "found": true}'
    service = _build_answering(tmp_path, FakeProvider(llm_json))
    result = service.answer("What was the total at Taco Bell?")
    assert isinstance(result, AnswerResponse)
    assert result.found is True
    assert "receipt:1" in result.citations


def test_answer_not_found_when_no_documents(tmp_path: Path):
    # Fresh db with no documents indexed
    repo = SqliteReceiptRepository(tmp_path / "empty.db")
    repo.init_db()
    retrieval = RetrievalService(
        SqliteRetrievalRepository(tmp_path / "empty.db"), HashEmbeddingProvider()
    )
    service = AnsweringService(FakeProvider('{"found": true}'), retrieval)
    result = service.answer("anything")
    assert result.found is False
    assert result.citations == []
    assert result == NOT_FOUND_ANSWER


def test_answer_not_found_on_bad_json(tmp_path: Path):
    _seed(tmp_path)
    service = _build_answering(tmp_path, FakeProvider("not json at all"))
    result = service.answer("Taco Bell total?")
    assert result.found is False
    assert result.citations == []


def test_answer_not_found_on_provider_error(tmp_path: Path):
    _seed(tmp_path)

    class FailingProvider(FakeProvider):
        def complete(self, messages, *, request_id=None, response_format=None):
            raise RuntimeError("provider down")

    service = _build_answering(tmp_path, FailingProvider(None))
    result = service.answer("Taco Bell total?")
    assert result.found is False


def test_answer_not_found_on_null_content(tmp_path: Path):
    _seed(tmp_path)
    service = _build_answering(tmp_path, FakeProvider(None))
    result = service.answer("Taco Bell total?")
    assert result.found is False


def test_answer_searches_receipt_documents_only():
    class CapturingRetrieval:
        def search(self, query, *, strategy="hybrid", limit=5, kind=None):
            self.kind = kind
            return []

    retrieval = CapturingRetrieval()
    service = AnsweringService(FakeProvider(None), retrieval)  # type: ignore[arg-type]
    service.answer("How much at Taco Bell?")
    assert retrieval.kind == "receipt"
