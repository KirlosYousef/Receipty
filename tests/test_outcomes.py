from types import SimpleNamespace

from app.domain.schemas import Outcome, ReceiptExtract
from app.services.extraction import ExtractionService
from app.services.postprocess import apply_postprocess


class FakeProvider:
    """Same pattern as tests/test_api.py — returns canned model content."""

    def __init__(self, content: str | None):
        self.content = content

    def complete(self, messages):
        return SimpleNamespace(
            model="fake",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0, model_extra={}),
        )


def test_malformed_model_json_is_extraction_failed():
    provider = FakeProvider("this is not json at all")
    row = ExtractionService(provider).extract_from_text("Apple Store TOTAL 1,234.56 USD")
    assert row.outcome == Outcome.extraction_failed
    assert row.merchant is None
    assert row.total is None


def test_complete_receipt_is_success():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Amazon",
        total="500.00",
        currency="USD",
        date="2024-01-15",
    )
    assert row.outcome == Outcome.success


def test_receipt_missing_total_needs_review():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Amazon",
        total=None,
        currency="USD",
    )
    assert row.outcome == Outcome.needs_review


def test_confident_non_receipt_scrubbed_and_classified():
    row = ReceiptExtract(
        is_receipt=False,
        merchant="Amazon",
        total="500.00",
        currency="USD",
    )
    out = apply_postprocess(row, "")
    assert out.outcome == Outcome.not_receipt
    assert out.merchant is None
    assert out.total is None
    assert out.currency is None

def test_unexpected_model_field_is_extraction_failed():
    provider = FakeProvider(
        """
        {
            "is_receipt": true,
            "merchant": "Amazon",
            "total": "50.00",
            "currency": "USD",
            "date": null,
            "tax": null,
            "unexpected": "should not be accepted"
        }
        """
    )

    row = ExtractionService(provider).extract_from_text(
        "Amazon Total 50.00 USD"
    )

    assert row.outcome == Outcome.extraction_failed
    assert row.merchant is None
    assert row.total is None

def test_answer_wrapper_is_extraction_failed():
    provider = FakeProvider(
        '{"answer": "{\\"is_receipt\\": false}"}'
    )

    row = ExtractionService(provider).extract_from_text("not a receipt")

    assert row.outcome == Outcome.extraction_failed

def test_missing_model_content_is_extraction_failed():
    provider = FakeProvider(None)

    row = ExtractionService(provider).extract_from_text("receipt")

    assert row.outcome == Outcome.extraction_failed
    assert row.total is None
