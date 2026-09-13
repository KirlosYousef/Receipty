import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.llm.provider import LLMProvider, OpenRouterProvider
from app.observability.usage import UsageLogger
from app.repository.receipts import ReceiptRepository
from app.services.extraction import ExtractionService

STATIC_DIR = Path(__file__).parent / "static"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

ProviderFactory = Callable[[Settings], LLMProvider]


def create_app(
    *,
    settings: Settings | None = None,
    provider_factory: ProviderFactory = OpenRouterProvider,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo = ReceiptRepository(resolved_settings.db_path)
        repo.init_db()

        provider = provider_factory(resolved_settings)
        usage = UsageLogger(resolved_settings.cost_log_path)
        service = ExtractionService(provider, usage)

        app.state.repo = repo
        app.state.extraction_service = service

        try:
            yield
        finally:
            provider.close()

    app = FastAPI(title="Receipty", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID")

        if incoming and REQUEST_ID_PATTERN.fullmatch(incoming):
            request_id = incoming
        else:
            request_id = str(uuid4())

        request.state.request_id = request_id

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
