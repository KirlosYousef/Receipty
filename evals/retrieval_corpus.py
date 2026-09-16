from __future__ import annotations

import json
from pathlib import Path

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import EmbeddingProvider
from app.repository.receipts import ReceiptRepository
from app.services.indexing import receipt_document_text

LABELS = Path("evals/labels.jsonl")
QUESTIONS = Path("evals/retrieval_questions.jsonl")


def source_id_for(file_name: str) -> str:
    return f"receipt:{file_name}"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def index_receipt_corpus(repo: ReceiptRepository, embeddings: EmbeddingProvider) -> int:
    count = 0
    for label in load_jsonl(LABELS):
        if not label.get("is_receipt"):
            continue
        extract = ReceiptExtract(
            is_receipt=True,
            merchant=label.get("merchant"),
            total=label.get("total"),
            currency=label.get("currency"),
            date=label.get("date"),
            outcome=Outcome.success,
        )
        source_id = source_id_for(label["file"])
        content = receipt_document_text(count + 1, extract)
        repo.save_document("receipt", source_id, content, embeddings.embed(content))
        count += 1
    return count
