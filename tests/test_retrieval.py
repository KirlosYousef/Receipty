from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.repository.receipts import SqliteReceiptRepository
from app.services.indexing import IndexingService
from app.services.retrieval import (
    RetrievalService,
    SqliteRetrievalRepository,
    cosine_similarity,
    hybrid_merge,
    keyword_terms,
    postgres_keyword_query,
)


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


def _build_retrieval(tmp_path: Path) -> RetrievalService:
    repo = SqliteRetrievalRepository(tmp_path / "receipts.db")
    return RetrievalService(repo, HashEmbeddingProvider())


def test_keyword_search_finds_matching_content(tmp_path: Path):
    _seed(tmp_path)
    retrieval = _build_retrieval(tmp_path)
    results = retrieval.search("Taco", strategy="keyword", limit=5)
    assert any("Taco Bell" in row["content"] for row in results)


def test_dense_search_returns_nearest_documents(tmp_path: Path):
    _seed(tmp_path)
    retrieval = _build_retrieval(tmp_path)
    results = retrieval.search("Taco Bell", strategy="dense", limit=5)
    assert len(results) <= 5
    assert any("Taco Bell" in row["content"] for row in results)


def test_hybrid_search_merges_keyword_and_dense(tmp_path: Path):
    _seed(tmp_path)
    retrieval = _build_retrieval(tmp_path)
    results = retrieval.search("Taco Bell total", strategy="hybrid", limit=5)
    assert len(results) <= 5
    source_ids = [row["source_id"] for row in results]
    assert len(source_ids) == len(set(source_ids))


def test_search_filters_by_kind(tmp_path: Path):
    _seed(tmp_path)
    retrieval = _build_retrieval(tmp_path)
    results = retrieval.search(
        "invent", strategy="keyword", limit=10, kind="policy_note"
    )
    assert results
    assert all(row["kind"] == "policy_note" for row in results)


def test_unknown_strategy_raises(tmp_path: Path):
    retrieval = _build_retrieval(tmp_path)
    with pytest.raises(ValueError, match="unknown strategy"):
        retrieval.search("x", strategy="invalid")


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_hybrid_merge_deduplicates_and_reranks():
    keyword = [
        {"source_id": "a", "content": "doc a", "kind": "receipt", "score": 1.0},
        {"source_id": "b", "content": "doc b", "kind": "receipt", "score": 0.5},
    ]
    dense = [
        {"source_id": "b", "content": "doc b", "kind": "receipt", "score": 0.9},
        {"source_id": "c", "content": "doc c", "kind": "receipt", "score": 0.8},
    ]
    merged = hybrid_merge(keyword, dense, limit=3)
    source_ids = [row["source_id"] for row in merged]
    assert set(source_ids) == {"a", "b", "c"}
    assert source_ids[0] == "b"


def test_postgres_keyword_query_binds_query_for_rank_and_match():
    sql, params = postgres_keyword_query("Amazon", limit=5, kind=None)
    assert sql.count("%s") == len(params)
    assert params == ["Amazon", "Amazon", 5]


def test_postgres_keyword_query_includes_kind_filter():
    sql, params = postgres_keyword_query("Amazon", limit=3, kind="receipt")
    assert sql.count("%s") == len(params)
    assert params == ["Amazon", "Amazon", "receipt", 3]
    assert "kind = %s" in sql


def test_keyword_terms_drops_question_stopwords():
    assert keyword_terms("How much did I spend at Amazon?") == ["Amazon"]


def test_keyword_search_matches_merchant_in_natural_question(tmp_path: Path):
    _seed(tmp_path)
    retrieval = _build_retrieval(tmp_path)
    results = retrieval.search(
        "How much did I spend at Taco Bell?",
        strategy="keyword",
        limit=5,
        kind="receipt",
    )
    assert any("Taco Bell" in row["content"] for row in results)
