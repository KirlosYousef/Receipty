from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.schemas import Outcome
from app.observability.tracing import SpanRecorder
from app.repository.receipts import ReceiptRepository
from app.services.retrieval import RetrievalService

LEDGER_QUERY_IDS = ("sum_total", "count", "totals_by_merchant")
LedgerQueryId = Literal["sum_total", "count", "totals_by_merchant"]
READ_TOOLS = frozenset({"search_receipts", "query_ledger"})
WRITE_TOOLS = frozenset({"flag_for_review", "mark_used"})
KNOWN_TOOLS = READ_TOOLS | WRITE_TOOLS


class ToolError(ValueError):
    """Caller passed a tool name or arguments we refuse to run."""


class SearchReceiptsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class QueryLedgerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: LedgerQueryId
    merchant: str | None = None


class ReceiptIdArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    receipt_id: int = Field(ge=1)


def _parameters(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_receipts",
            "description": (
                "Search indexed receipts by merchant, date, or free text. "
                "Returns ranked snippets with source_id."
            ),
            "parameters": _parameters(SearchReceiptsArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_ledger",
            "description": (
                "Run a named ledger aggregate. query_id must be sum_total, "
                "count, or totals_by_merchant. Optional merchant filter. "
                "Does not accept SQL."
            ),
            "parameters": _parameters(QueryLedgerArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_review",
            "description": (
                "Propose setting a receipt outcome to needs_review. "
                "This write needs approval and will not run immediately."
            ),
            "parameters": _parameters(ReceiptIdArgs),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_used",
            "description": (
                "Propose marking a receipt as used. "
                "This write needs approval and will not run immediately."
            ),
            "parameters": _parameters(ReceiptIdArgs),
        },
    },
]


def parse_args(model: type[BaseModel], raw: dict[str, Any]) -> BaseModel:
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ToolError(str(exc)) from exc


class AgentTools:
    def __init__(
        self,
        repo: ReceiptRepository,
        retrieval: RetrievalService,
        *,
        spans: SpanRecorder | None = None,
    ):
        self._repo = repo
        self._retrieval = retrieval
        self._spans = spans or SpanRecorder(None)

    def search_receipts(self, raw: dict[str, Any]) -> list[dict]:
        args = parse_args(SearchReceiptsArgs, raw)
        assert isinstance(args, SearchReceiptsArgs)
        return self._retrieval.search(
            args.q, strategy="hybrid", limit=args.limit, kind="receipt"
        )

    def query_ledger(self, raw: dict[str, Any]) -> dict[str, Any]:
        args = parse_args(QueryLedgerArgs, raw)
        assert isinstance(args, QueryLedgerArgs)
        if args.query_id == "sum_total":
            result = self._repo.ledger_sum_total(args.merchant)
        elif args.query_id == "count":
            result = self._repo.ledger_count(args.merchant)
        else:
            result = {"rows": self._repo.ledger_totals_by_merchant()}
        return {"query_id": args.query_id, "merchant": args.merchant, **result}

    def flag_for_review(self, raw: dict[str, Any]) -> dict:
        args = parse_args(ReceiptIdArgs, raw)
        assert isinstance(args, ReceiptIdArgs)
        try:
            row = self._repo.update_outcome(args.receipt_id, Outcome.needs_review.value)
        except LookupError as exc:
            raise ToolError(str(exc)) from exc
        return row

    def mark_used(self, raw: dict[str, Any]) -> dict:
        args = parse_args(ReceiptIdArgs, raw)
        assert isinstance(args, ReceiptIdArgs)
        try:
            row = self._repo.set_used(args.receipt_id, True)
        except LookupError as exc:
            raise ToolError(str(exc)) from exc
        return row

    def dispatch(self, name: str, raw: dict[str, Any]) -> Any:
        with self._spans.span(
            f"execute_tool {name}",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name},
        ):
            if name not in KNOWN_TOOLS:
                raise ToolError(f"unknown tool: {name}")
            return getattr(self, name)(raw)
