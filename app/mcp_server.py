from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.core.config import Settings, get_settings
from app.llm.embeddings import CachingEmbeddings, OpenRouterEmbeddings
from app.repository.receipts import build_repository
from app.services.retrieval import RetrievalService, build_retrieval_repository
from app.services.tools import AgentTools, LedgerQueryId

INSTRUCTIONS = (
    "Receipt ledger tools. search_receipts and query_ledger only read. "
    "query_ledger accepts query_id sum_total, count, or totals_by_merchant. "
    "It does not accept SQL. flag_for_review and mark_used change a row "
    "when you call them. The HTTP agent pauses those writes; this server does not."
)


def create_server(tools: AgentTools) -> FastMCP:
    mcp = FastMCP("receipty", instructions=INSTRUCTIONS)

    @mcp.tool(
        description=(
            "Search indexed receipts by merchant, date, or free text. "
            "Returns ranked snippets with source_id."
        )
    )
    def search_receipts(
        q: Annotated[str, Field(min_length=1)],
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> list[dict[str, Any]]:
        return tools.search_receipts({"q": q, "limit": limit})

    @mcp.tool(
        description=(
            "Run a named ledger aggregate. query_id must be sum_total, "
            "count, or totals_by_merchant. Optional merchant filter. "
            "Does not accept SQL."
        )
    )
    def query_ledger(
        query_id: LedgerQueryId,
        merchant: str | None = None,
    ) -> dict[str, Any]:
        return tools.query_ledger({"query_id": query_id, "merchant": merchant})

    @mcp.tool(
        description="Set a receipt outcome to needs_review. This write runs immediately."
    )
    def flag_for_review(receipt_id: Annotated[int, Field(ge=1)]) -> dict[str, Any]:
        return tools.flag_for_review({"receipt_id": receipt_id})

    @mcp.tool(description="Mark a receipt as used. This write runs immediately.")
    def mark_used(receipt_id: Annotated[int, Field(ge=1)]) -> dict[str, Any]:
        return tools.mark_used({"receipt_id": receipt_id})

    return mcp


def build_server(settings: Settings | None = None) -> FastMCP:
    resolved = settings or get_settings()
    repo = build_repository(
        db_path=resolved.db_path,
        database_url=resolved.database_url,
    )
    repo.init_db()
    retrieval = RetrievalService(
        build_retrieval_repository(
            db_path=resolved.db_path,
            database_url=resolved.database_url,
        ),
        CachingEmbeddings(
            OpenRouterEmbeddings(resolved),
            max_entries=resolved.embedding_cache_size,
        ),
    )
    return create_server(AgentTools(repo, retrieval))


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
