from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.repository.receipts import SqliteReceiptRepository
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository
from app.services.tools import AgentTools, ToolError


def _tools(tmp_path: Path) -> tuple[AgentTools, int]:
    db_path = tmp_path / "receipts.db"
    repo = SqliteReceiptRepository(db_path)
    repo.init_db()
    extract = ReceiptExtract(
        is_receipt=True,
        merchant="Taco Bell",
        total="7.61",
        currency="USD",
        date="2016-09-01",
        outcome=Outcome.success,
    )
    receipt_id = repo.save(extract)
    indexer = IndexingService(repo, HashEmbeddingProvider())
    indexer.index_receipt(receipt_id, extract)
    retrieval = RetrievalService(
        SqliteRetrievalRepository(db_path), HashEmbeddingProvider()
    )
    return AgentTools(repo, retrieval), receipt_id


def test_search_receipts_returns_indexed_merchant(tmp_path: Path):
    tools, _receipt_id = _tools(tmp_path)
    hits = tools.search_receipts({"q": "Taco Bell", "limit": 5})
    assert any("Taco Bell" in hit["content"] for hit in hits)


def test_query_ledger_sum_and_count(tmp_path: Path):
    tools, _receipt_id = _tools(tmp_path)
    total = tools.query_ledger({"query_id": "sum_total", "merchant": "Taco Bell"})
    assert total["n"] == 1
    assert float(total["amount"]) == pytest.approx(7.61)
    count = tools.query_ledger({"query_id": "count"})
    assert count["n"] == 1
    grouped = tools.query_ledger({"query_id": "totals_by_merchant"})
    assert grouped["rows"][0]["merchant"] == "Taco Bell"


def test_query_ledger_rejects_sql_and_unknown_names(tmp_path: Path):
    tools, _receipt_id = _tools(tmp_path)
    with pytest.raises(ToolError):
        tools.query_ledger({"query_id": "DROP TABLE receipts"})
    with pytest.raises(ToolError):
        tools.query_ledger({"sql": "SELECT * FROM receipts"})


def test_flag_for_review_and_mark_used(tmp_path: Path):
    tools, receipt_id = _tools(tmp_path)
    flagged = tools.flag_for_review({"receipt_id": receipt_id})
    assert flagged["outcome"] == Outcome.needs_review.value
    used = tools.mark_used({"receipt_id": receipt_id})
    assert int(used["used"]) == 1


def test_mutations_fail_for_missing_receipt(tmp_path: Path):
    tools, _receipt_id = _tools(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        tools.flag_for_review({"receipt_id": 999})
    with pytest.raises(ToolError, match="not found"):
        tools.mark_used({"receipt_id": 999})
