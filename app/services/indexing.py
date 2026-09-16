from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.schemas import Outcome, ReceiptExtract
from app.llm.embeddings import EmbeddingProvider
from app.repository.receipts import ReceiptRepository

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "seed"
ALIASES_PATH = DATA_DIR / "merchant_aliases.json"
NOTES_PATH = DATA_DIR / "policy_notes.json"


def receipt_document_text(receipt_id: int, row: ReceiptExtract) -> str:
    merchant = row.merchant or "unknown"
    total = "null" if row.total is None else str(row.total)
    currency = row.currency or "null"
    date = "null" if row.date is None else str(row.date)
    outcome = row.outcome.value if row.outcome is not None else "null"
    return (
        f"Receipt {receipt_id}. Merchant: {merchant}. "
        f"Total: {total} {currency}. Date: {date}. Outcome: {outcome}."
    )


def extract_from_row(row: dict) -> ReceiptExtract:
    return ReceiptExtract(
        is_receipt=bool(row["is_receipt"]),
        merchant=row.get("merchant"),
        total=row.get("total"),
        currency=row.get("currency"),
        date=row.get("date"),
        tax=row.get("tax"),
        outcome=row.get("outcome"),
    )


class IndexingService:
    def __init__(self, repo: ReceiptRepository, embeddings: EmbeddingProvider):
        self._repo = repo
        self._embeddings = embeddings

    def seed_static_documents(self) -> None:
        existing = {row["source_id"] for row in self._repo.list_documents()}
        aliases = json.loads(ALIASES_PATH.read_text())
        for alias in aliases:
            source_id = f"alias:{alias['alias']}"
            if source_id in existing:
                continue
            content = (
                f"Merchant alias: {alias['alias']} refers to {alias['canonical']}."
            )
            self._index("merchant_alias", source_id, content)

        notes = json.loads(NOTES_PATH.read_text())
        for note in notes:
            source_id = f"policy:{note['id']}"
            if source_id in existing:
                continue
            self._index("policy_note", source_id, note["content"])

    def index_existing_receipts(self) -> None:
        existing = {row["source_id"] for row in self._repo.list_documents()}
        for row in self._repo.list_all():
            receipt_id = int(row["id"])
            if f"receipt:{receipt_id}" in existing:
                continue
            try:
                self.index_receipt(receipt_id, extract_from_row(row))
            except Exception:
                log.exception("reindex_failed receipt_id=%s", receipt_id)

    def index_receipt(self, receipt_id: int, row: ReceiptExtract) -> None:
        if row.outcome in {Outcome.not_receipt, Outcome.extraction_failed}:
            return
        source_id = f"receipt:{receipt_id}"
        self._index("receipt", source_id, receipt_document_text(receipt_id, row))

    def _index(self, kind: str, source_id: str, content: str) -> None:
        embedding = self._embeddings.embed(content)
        self._repo.save_document(kind, source_id, content, embedding)
