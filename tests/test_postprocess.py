from decimal import Decimal

from app.domain.schemas import ReceiptExtract
from app.services.postprocess import apply_postprocess, infer_currency


def test_infer_egp():
    assert infer_currency("Carrefour\nTOTAL 186.50 EGP") == "EGP"


def test_infer_ambiguous_returns_none():
    assert infer_currency("paid $10 and €5") is None


def test_non_receipt_nulls_money_fields():
    row = ReceiptExtract(
        is_receipt=False,
        merchant="Nope",
        total=Decimal("10.00"),
        currency="USD",
        date=None,
        tax=Decimal("1.00"),
        needs_review=False,
    )
    out = apply_postprocess(row, "EGP")
    assert out.merchant is None
    assert out.total is None
    assert out.currency is None
    assert out.tax is None
    assert out.needs_review is True


def test_missing_total_forces_review():
    row = ReceiptExtract(is_receipt=True, merchant="Shop", total=None, currency="USD")
    out = apply_postprocess(row, "")
    assert out.needs_review is True


def test_currency_inferred_when_missing():
    row = ReceiptExtract(is_receipt=True, merchant="Shop", total=Decimal("1.00"), currency=None)
    out = apply_postprocess(row, "TOTAL 1.00 EGP")
    assert out.currency == "EGP"
