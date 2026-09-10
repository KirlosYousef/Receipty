from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.domain.schemas import Outcome
from app.llm.provider import LLMProvider, OpenRouterProvider
from app.observability.usage import UsageLogger
from app.repository.receipts import ReceiptRepository
from app.services.extraction import ExtractionService

STATIC_DIR = Path(__file__).parent / "static"


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
    app.include_router(router)

    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    
    return app


app = create_app()
