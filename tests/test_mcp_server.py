from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import HashEmbeddingProvider
from app.mcp_server import create_server
from app.repository.receipts import SqliteReceiptRepository
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, SqliteRetrievalRepository
from app.services.tools import AgentTools


def _tools(tmp_path: Path) -> tuple[AgentTools, SqliteReceiptRepository, int]:
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
    IndexingService(repo, HashEmbeddingProvider()).index_receipt(receipt_id, extract)
    tools = AgentTools(
        repo,
        RetrievalService(SqliteRetrievalRepository(db_path), HashEmbeddingProvider()),
    )
    return tools, repo, receipt_id


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


def test_mcp_lists_the_same_four_tools(tmp_path: Path):
    tools, _repo, _receipt_id = _tools(tmp_path)

    async def scenario():
        server = create_server(tools)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            listed = await session.list_tools()
            return {tool.name for tool in listed.tools}

    names = asyncio.run(scenario())
    assert names == {
        "search_receipts",
        "query_ledger",
        "flag_for_review",
        "mark_used",
    }


def test_mcp_query_ledger_reads_and_rejects_sql(tmp_path: Path):
    tools, _repo, _receipt_id = _tools(tmp_path)

    async def scenario():
        server = create_server(tools)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            total = await session.call_tool(
                "query_ledger",
                {"query_id": "sum_total", "merchant": "Taco Bell"},
            )
            rejected = await session.call_tool(
                "query_ledger",
                {"query_id": "DROP TABLE receipts", "sql": "SELECT * FROM receipts"},
            )
            return total, rejected

    total, rejected = asyncio.run(scenario())
    assert total.isError is False
    body = _payload(total)
    assert float(body["amount"]) == pytest.approx(7.61)
    assert rejected.isError is True
    assert "DROP TABLE" in rejected.content[0].text


def test_mcp_mark_used_writes_immediately(tmp_path: Path):
    tools, repo, receipt_id = _tools(tmp_path)

    async def scenario():
        server = create_server(tools)
        async with create_connected_server_and_client_session(server) as session:
            await session.initialize()
            return await session.call_tool("mark_used", {"receipt_id": receipt_id})

    result = asyncio.run(scenario())
    assert result.isError is False
    assert int(repo.get(receipt_id)["used"]) == 1
