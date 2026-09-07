from decimal import Decimal

from app.domain.schemas import ReceiptExtract
from evals.scoring import receipt_hit, score_row, total_hit


def test_receipt_hit():
    pred = ReceiptExtract(is_receipt=True)
    assert receipt_hit(pred, {"is_receipt": True})
    assert not receipt_hit(pred, {"is_receipt": False})


def test_total_both_null():
    pred = ReceiptExtract(is_receipt=True, total=None)
    assert total_hit(pred, {"total": None})


def test_total_mismatch():
    pred = ReceiptExtract(is_receipt=True, total=Decimal("10.00"))
    assert not total_hit(pred, {"total": 9.99})


def test_total_match():
    pred = ReceiptExtract(is_receipt=True, total=Decimal("28.31"))
    assert total_hit(pred, {"total": 28.31})


def test_score_row_ok():
    pred = ReceiptExtract(is_receipt=True, merchant="X", total=Decimal("1.00"))
    scored = score_row(pred, {"is_receipt": True, "merchant": "Y", "total": 1.0})
    assert scored["ok"] is True
    assert scored["receipt_ok"] is True
    assert scored["total_ok"] is True
