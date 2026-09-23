import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.rate_limit import LIMITED_ROUTES, RateLimiter
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.llm.embeddings import (
    CachingEmbeddings,
    EmbeddingProvider,
    OpenRouterEmbeddings,
)
from app.llm.provider import LLMProvider, OpenRouterProvider
from app.observability.tracing import SpanRecorder, bind_request_id
from app.observability.usage import UsageLogger
from app.repository.receipts import build_repository
from app.services.agent import AgentService
from app.services.answering import AnsweringService
from app.services.extraction import ExtractionService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService, build_retrieval_repository
from app.services.tools import AgentTools

STATIC_DIR = Path(__file__).parent / "static"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

ProviderFactory = Callable[[Settings], LLMProvider]
EmbeddingFactory = Callable[[Settings], EmbeddingProvider]


def create_app(
    *,
    settings: Settings | None = None,
    provider_factory: ProviderFactory = OpenRouterProvider,
    embedding_factory: EmbeddingFactory = OpenRouterEmbeddings,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo = build_repository(
            db_path=resolved_settings.db_path,
            database_url=resolved_settings.database_url,
        )
        repo.init_db()

        spans = SpanRecorder(resolved_settings.trace_log_path)
        if provider_factory is OpenRouterProvider:
            provider = OpenRouterProvider(resolved_settings, spans=spans)
        else:
            provider = provider_factory(resolved_settings)
        if embedding_factory is OpenRouterEmbeddings:
            embeddings = OpenRouterEmbeddings(resolved_settings, spans=spans)
        else:
            embeddings = embedding_factory(resolved_settings)
        embeddings = CachingEmbeddings(
            embeddings,
            max_entries=resolved_settings.embedding_cache_size,
        )
        usage = UsageLogger(resolved_settings.cost_log_path)
        service = ExtractionService(provider, usage)
        indexer = IndexingService(repo, embeddings)
        indexer.seed_static_documents()
        indexer.index_existing_receipts()
        retrieval_repo = build_retrieval_repository(
            db_path=resolved_settings.db_path,
            database_url=resolved_settings.database_url,
        )
        retrieval_service = RetrievalService(retrieval_repo, embeddings, spans=spans)
        answering_service = AnsweringService(provider, retrieval_service)
        agent_service = AgentService(
            provider,
            AgentTools(repo, retrieval_service, spans=spans),
            max_steps=resolved_settings.max_agent_steps,
            deadline_seconds=resolved_settings.max_agent_seconds,
            token_budget=resolved_settings.max_agent_tokens,
        )

        app.state.repo = repo
        app.state.extraction_service = service
        app.state.indexer = indexer
        app.state.retrieval_service = retrieval_service
        app.state.answering_service = answering_service
        app.state.agent_service = agent_service

        try:
            yield
        finally:
            provider.close()

    app = FastAPI(title="Receipty", lifespan=lifespan)
    limiter = RateLimiter(
        resolved_settings.rate_limit_requests,
        resolved_settings.rate_limit_window_seconds,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID")

        if incoming and REQUEST_ID_PATTERN.fullmatch(incoming):
            request_id = incoming
        else:
            request_id = str(uuid4())

        request.state.request_id = request_id
        client = request.client.host if request.client else "unknown"
        route = (request.method, request.url.path)
        if route in LIMITED_ROUTES and not limiter.allow(client):
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "Retry-After": str(int(resolved_settings.rate_limit_window_seconds))
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response

        with bind_request_id(request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(router)

    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
