from datetime import date as Date
from datetime import datetime
from decimal import Decimal
from enum import Enum
import re

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_validator, model_validator

class Outcome(str, Enum):
    success = "success"
    needs_review = "needs_review"
    not_receipt = "not_receipt"
    extraction_failed = "extraction_failed"


def _money(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float, Decimal)):
        return v
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(v))
    if not m:
        return None
    return m.group(0).replace(",", ".")


class ReceiptExtract(BaseModel):
    is_receipt: bool
    merchant: str | None = None
    total: Decimal | None = Field(default=None, decimal_places=2)
    currency: str | None = None
    date: Date | None = None
    tax: Decimal | None = Field(default=None, decimal_places=2)
    outcome: Outcome | None = None
    _needs_review_override: bool | None = PrivateAttr(default=None)

    @computed_field
    @property
    def needs_review(self) -> bool:
        if self._needs_review_override is not None:
            return self._needs_review_override
        return self.outcome in {Outcome.needs_review, Outcome.extraction_failed}

    @needs_review.setter
    def needs_review(self, value: bool) -> None:
        self._needs_review_override = value
    
    @field_validator("total", "tax", mode="before")
    @classmethod
    def money_number_only(cls, v):
        return _money(v)

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, Date):
            return v
        s = str(v).strip()
        for fmt in (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%b %d '%y",
            "%b %d, %Y",
            "%B %d, %Y",
        ):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    @model_validator(mode="after")
    def _resolve_outcome(self) -> "ReceiptExtract":
        if self.outcome is None:
            if not self.is_receipt:
                self.outcome = Outcome.not_receipt
            elif self.total is None or self.currency is None:
                self.outcome = Outcome.needs_review
            else:
                self.outcome = Outcome.success
        return self



class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    extract: ReceiptExtract