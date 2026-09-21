from functools import lru_cache

from fastapi import Request

from app.core.config import get_settings
from app.observability.usage import UsageLogger
from app.repository.receipts import ReceiptRepository
from app.services.agent import AgentService
from app.services.answering import AnsweringService
from app.services.extraction import ExtractionService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService

__all__ = [
    "get_settings",
    "get_repo",
    "get_extraction_service",
    "get_indexer",
    "get_retrieval_service",
    "get_answering_service",
    "get_agent_service",
]


def get_repo(request: Request) -> ReceiptRepository:
    return request.app.state.repo


@lru_cache
def _usage_logger(path: str) -> UsageLogger:
    from pathlib import Path

    return UsageLogger(Path(path))


def get_extraction_service(request: Request) -> ExtractionService:
    return request.app.state.extraction_service


def get_indexer(request: Request) -> IndexingService:
    return request.app.state.indexer


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service


def get_answering_service(request: Request) -> AnsweringService:
    return request.app.state.answering_service


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service
