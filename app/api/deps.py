from functools import lru_cache

from fastapi import Request

from app.core.config import get_settings
from app.observability.usage import UsageLogger
from app.repository.receipts import ReceiptRepository
from app.services.extraction import ExtractionService

__all__ = [
    "get_settings",
    "get_repo",
    "get_extraction_service",
]


def get_repo(request: Request) -> ReceiptRepository:
    return request.app.state.repo


@lru_cache
def _usage_logger(path: str) -> UsageLogger:
    from pathlib import Path

    return UsageLogger(Path(path))


def get_extraction_service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service
