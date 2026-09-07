from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_extraction_service, get_repo, get_settings
from app.core.config import Settings
from app.core.exceptions import CreditsExhausted, DailyLimitReached, ProviderError
from app.domain.schemas import IngestResponse, IngestTextRequest
from app.repository.receipts import ReceiptRepository
from app.services.extraction import ExtractionService

router = APIRouter()

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def _map_provider_error(exc: ProviderError) -> HTTPException:
    if isinstance(exc, CreditsExhausted):
        return HTTPException(status_code=402, detail=str(exc))
    if isinstance(exc, DailyLimitReached):
        return HTTPException(status_code=429, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/ingest", response_model=IngestResponse)
def ingest(
    req: IngestTextRequest,
    service: ExtractionService = Depends(get_extraction_service),
    repo: ReceiptRepository = Depends(get_repo),
) -> IngestResponse:
    try:
        extract = service.extract_from_text(req.text)
    except ProviderError as e:
        raise _map_provider_error(e) from e
    repo.save(extract)
    return IngestResponse(extract=extract)


@router.post("/v1/ingest/image", response_model=IngestResponse)
def ingest_image(
    file: UploadFile = File(...),
    service: ExtractionService = Depends(get_extraction_service),
    repo: ReceiptRepository = Depends(get_repo),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    mime = file.content_type or "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, "jpeg/png/webp only")
    data = file.file.read()
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"image exceeds {settings.max_image_bytes} bytes")
    try:
        extract = service.extract_from_image(data, mime)
    except ProviderError as e:
        raise _map_provider_error(e) from e
    repo.save(extract)
    return IngestResponse(extract=extract)


@router.get("/v1/receipts")
def receipts(repo: ReceiptRepository = Depends(get_repo)) -> list[dict]:
    return repo.list_all()
