from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.domain.schemas import Outcome
from app.repository.receipts import ReceiptRepository

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repo = ReceiptRepository(settings.db_path)
    repo.init_db()
    app.state.repo = repo
    yield


def create_app() -> FastAPI:
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
