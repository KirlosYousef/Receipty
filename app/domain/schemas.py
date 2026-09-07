from datetime import date as Date
from datetime import datetime
from decimal import Decimal
import re

from pydantic import BaseModel, Field, field_validator


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
    needs_review: bool = False

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


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    extract: ReceiptExtract
