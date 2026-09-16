import json
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import (
    get_answering_service,
    get_extraction_service,
    get_indexer,
    get_repo,
    get_retrieval_service,
    get_settings,
)
from app.core.config import Settings
from app.core.exceptions import (
    CreditsExhausted,
    DailyLimitReached,
    ProviderDeadlineExceeded,
    ProviderError,
)
from app.domain.schemas import (
    AnswerResponse,
    AskRequest,
    IngestResponse,
    IngestTextRequest,
    ReceiptExtract,
)
from app.repository.receipts import ReceiptRepository
from app.services.answering import AnsweringService
from app.services.extraction import ExtractionService
from app.services.indexing import IndexingService
from app.services.retrieval import RetrievalService

router = APIRouter()
log = logging.getLogger(__name__)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def _index_receipt(
    indexer: IndexingService, receipt_id: int, extract: ReceiptExtract
) -> None:
    try:
        indexer.index_receipt(receipt_id, extract)
    except Exception:
        log.exception("indexing_failed receipt_id=%s", receipt_id)


def _map_provider_error(exc: ProviderError) -> HTTPException:
    if isinstance(exc, ProviderDeadlineExceeded):
        return HTTPException(status_code=504, detail=str(exc))
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
    request: Request,
    service: ExtractionService = Depends(get_extraction_service),
    repo: ReceiptRepository = Depends(get_repo),
    indexer: IndexingService = Depends(get_indexer),
) -> IngestResponse:
    try:
        extract = service.extract_from_text(
            req.text,
            request_id=request.state.request_id,
        )
    except ProviderError as e:
        raise _map_provider_error(e) from e
    receipt_id = repo.save(extract)
    _index_receipt(indexer, receipt_id, extract)
    return IngestResponse(extract=extract)


@router.post("/v1/ingest/image", response_model=IngestResponse)
def ingest_image(
    request: Request,
    file: UploadFile = File(...),
    service: ExtractionService = Depends(get_extraction_service),
    repo: ReceiptRepository = Depends(get_repo),
    indexer: IndexingService = Depends(get_indexer),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    mime = file.content_type or "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, "jpeg/png/webp only")
    data = file.file.read()
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, f"image exceeds {settings.max_image_bytes} bytes")
    try:
        extract = service.extract_from_image(
            data,
            mime,
            request_id=request.state.request_id,
        )
    except ProviderError as e:
        raise _map_provider_error(e) from e
    receipt_id = repo.save(extract)
    _index_receipt(indexer, receipt_id, extract)
    return IngestResponse(extract=extract)


@router.get("/v1/receipts")
def receipts(repo: ReceiptRepository = Depends(get_repo)) -> list[dict]:
    return repo.list_all()


@router.get("/v1/search")
def search(
    q: str,
    strategy: str = "hybrid",
    limit: int = 5,
    kind: str | None = None,
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> list[dict]:
    if not q.strip():
        raise HTTPException(400, "query must not be empty")
    if strategy not in {"keyword", "dense", "hybrid"}:
        raise HTTPException(400, "strategy must be keyword, dense, or hybrid")
    if limit < 1 or limit > 50:
        raise HTTPException(400, "limit must be between 1 and 50")
    return retrieval.search(q, strategy=strategy, limit=limit, kind=kind)


@router.post("/v1/ask", response_model=AnswerResponse)
def ask(
    req: AskRequest,
    request: Request,
    answering: AnsweringService = Depends(get_answering_service),
) -> AnswerResponse:
    if req.strategy not in {"keyword", "dense", "hybrid"}:
        raise HTTPException(400, "strategy must be keyword, dense, or hybrid")
    if req.limit < 1 or req.limit > 20:
        raise HTTPException(400, "limit must be between 1 and 20")
    return answering.answer(req.question, request_id=request.state.request_id)


@router.get("/v1/usage")
def usage(settings: Settings = Depends(get_settings)) -> dict:
    """Return aggregated cost/token usage from the JSONL log."""
    log_path = settings.cost_log_path
    if not log_path.exists():
        return {
            "total_calls": 0,
            "total_usd": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "calls": [],
        }

    calls: list[dict] = []
    total_usd = 0.0
    total_prompt = 0
    total_completion = 0

    for line in log_path.read_text().strip().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls.append(entry)
        cost = entry.get("usd")
        if cost is not None:
            total_usd += float(cost)
        pt = entry.get("prompt_tokens")
        if pt is not None:
            total_prompt += int(pt)
        ct = entry.get("completion_tokens")
        if ct is not None:
            total_completion += int(ct)

    avg_usd = total_usd / len(calls) if calls else 0

    return {
        "total_calls": len(calls),
        "total_usd": round(total_usd, 6),
        "avg_usd_per_call": round(avg_usd, 6),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "calls": calls[-50:],  # last 50 entries
    }
