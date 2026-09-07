from datetime import date
from decimal import Decimal

from app.domain.schemas import ReceiptExtract


def test_total_strips_currency():
    row = ReceiptExtract(is_receipt=True, total="186.50 EGP", needs_review=False)
    assert row.total == Decimal("186.50")


def test_unreadable_total_needs_review_flag_on_model():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Carrefour",
        total=None,
        currency="EGP",
        needs_review=True,
    )
    assert row.total is None
    assert row.needs_review is True


def test_not_a_receipt_defaults():
    row = ReceiptExtract(is_receipt=False)
    assert row.merchant is None
    assert row.total is None


def test_valid_receipt():
    row = ReceiptExtract(
        is_receipt=True,
        merchant="Amazon",
        total=500.0,
        currency="USD",
        needs_review=False,
    )
    assert row.total == Decimal("500.0")
    assert row.merchant == "Amazon"
    assert row.currency == "USD"


def test_date_formats():
    row = ReceiptExtract(is_receipt=True, date="2017-11-24")
    assert row.date == date(2017, 11, 24)
    row2 = ReceiptExtract(is_receipt=True, date="11/24/2017")
    assert row2.date == date(2017, 11, 24)


def test_bad_date_becomes_none():
    row = ReceiptExtract(is_receipt=True, date="not-a-date")
    assert row.date is None
