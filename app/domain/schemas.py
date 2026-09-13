import re
from datetime import date as Date
from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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

    match = re.search(r"-?\d[\d.,]*", str(v))
    if not match:
        return None

    token = match.group(0)

    if "." in token and "," in token:
        if token.rfind(".") > token.rfind(","):
            # 1,234.56 → 1234.56
            token = token.replace(",", "")
        else:
            # 1.234,56 → 1234.56
            token = token.replace(".", "").replace(",", ".")

    # A single separator followed by exactly three digits is ambiguous:
    # "1.234" could mean 1.234 or 1,234.
    if token.count(".") + token.count(",") == 1:
        separator = "." if "." in token else ","
        fractional_part = token.rsplit(separator, 1)[1]

        if len(fractional_part) == 3:
            return None

    elif "," in token:
        # 12,50 → 12.50
        token = token.replace(",", ".")

    return token


class ReceiptLLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_receipt: bool
    merchant: str | None
    total: str | None
    currency: str | None
    date: str | None
    tax: str | None


class ReceiptExtract(BaseModel):
    is_receipt: bool
    merchant: str | None = None
    total: Decimal | None = Field(default=None, decimal_places=2)
    currency: str | None = None
    date: Date | None = None
    tax: Decimal | None = Field(default=None, decimal_places=2)
    outcome: Outcome | None = None

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
            elif self.total is None or self.currency is None or self.total < 0:
                self.outcome = Outcome.needs_review
            else:
                self.outcome = Outcome.success
        return self


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    extract: ReceiptExtract
