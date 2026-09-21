from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.schemas import Outcome
from app.repository.receipts import ReceiptRepository
from app.services.retrieval import RetrievalService

LEDGER_QUERY_IDS = ("sum_total", "count", "totals_by_merchant")
LedgerQueryId = Literal["sum_total", "count", "totals_by_merchant"]


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


def parse_args(model: type[BaseModel], raw: dict[str, Any]) -> BaseModel:
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ToolError(str(exc)) from exc


class AgentTools:
    def __init__(self, repo: ReceiptRepository, retrieval: RetrievalService):
        self._repo = repo
        self._retrieval = retrieval

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
