from decimal import Decimal

from app.domain.schemas import Outcome, ReceiptExtract
from app.services.extraction import ExtractionService
from tests.test_outcomes import FakeProvider


def test_parses_us_thousands_separator():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Apple Store",
        total="1,234.56",
        currency="USD",
    )

    assert row.total == Decimal("1234.56")
    assert row.outcome == Outcome.success


def test_parses_european_thousands_separator():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="European Store",
        total="1.234,56",
        currency="EUR",
    )

    assert row.total == Decimal("1234.56")
    assert row.outcome == Outcome.success


def test_parses_simple_decimal():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Shop",
        total="12.50",
        currency="USD",
    )

    assert row.total == Decimal("12.50")
    assert row.outcome == Outcome.success


def test_negative_total_needs_review():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Shop",
        total="-50.00",
        currency="USD",
    )

    assert row.total == Decimal("-50.00")
    assert row.outcome == Outcome.needs_review


def test_ambiguous_single_separator_needs_review():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Shop",
        total="1.234",
        currency="EUR",
    )

    assert row.total is None
    assert row.outcome == Outcome.needs_review


def test_apple_store_total_does_not_become_extraction_failure():
    provider = FakeProvider(
        """
        {
            "is_receipt": true,
            "merchant": "Apple Store",
            "total": "1,234.56",
            "currency": "USD",
            "date": null,
            "tax": null
        }
        """
    )

    row = ExtractionService(provider).extract_from_text(
        "Apple Store Total 1,234.56 USD"
    )

    assert row.total == Decimal("1234.56")
    assert row.outcome == Outcome.success
