from decimal import Decimal

from app.domain.schemas import Outcome, ReceiptExtract
from evals.scoring import (
    hallucinated_date,
    hallucinated_total,
    receipt_hit,
    score_row,
    summarize_rows,
    total_hit,
)


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


def test_pred_date_serialized_when_total_is_null():
    """pred_date must come from pred.date when total is absent."""
    pred = ReceiptExtract(
        is_receipt=True,
        merchant="the golden pear cafe",
        total=None,
        currency=None,
        date="2015-08-06",
    )
    scored = score_row(
        pred,
        {
            "is_receipt": True,
            "merchant": "the golden pear Cafe",
            "total": None,
            "currency": None,
            "date": "2015-08-06",
        },
    )
    assert scored["pred_total"] is None
    assert scored["pred_date"] == "2015-08-06"
    assert scored["date_ok"] is True
    assert scored["total_ok"] is True
    assert scored["ok"] is True
    assert scored["hallucinated_total"] is False


def test_hallucinated_total_on_gold_null():
    pred = ReceiptExtract(is_receipt=True, total=Decimal("12.00"), currency="USD")
    gold = {"is_receipt": True, "total": None, "currency": None, "date": "2015-08-06"}
    assert hallucinated_total(pred, gold)
    assert not hallucinated_date(pred, gold)
    scored = score_row(pred, gold)
    assert scored["hallucinated_total"] is True
    assert scored["total_ok"] is False


def test_non_receipt_null_fields_are_not_hallucinations():
    pred = ReceiptExtract(is_receipt=False, outcome=Outcome.not_receipt)
    gold = {"is_receipt": False, "total": None, "currency": None, "date": None}
    assert not hallucinated_total(pred, gold)
    assert not hallucinated_date(pred, gold)


def test_hallucinated_date_on_gold_null():
    pred = ReceiptExtract(
        is_receipt=True,
        total=Decimal("10.00"),
        currency="USD",
        date="2019-01-01",
    )
    gold = {"is_receipt": True, "total": 10.0, "currency": "USD", "date": None}
    assert hallucinated_date(pred, gold)
    scored = score_row(pred, gold)
    assert scored["hallucinated_date"] is True
    assert scored["date_ok"] is False


def test_both_null_is_not_hallucination():
    pred = ReceiptExtract(is_receipt=True, total=None, currency=None, date=None)
    gold = {"is_receipt": True, "total": None, "currency": None, "date": None}
    assert not hallucinated_total(pred, gold)
    assert not hallucinated_date(pred, gold)


def test_summarize_rows_includes_safety_metrics():
    rows = [
        {
            "receipt_ok": True,
            "total_ok": True,
            "date_ok": True,
            "ok": True,
            "hallucinated_total": False,
            "hallucinated_date": False,
            "gold_total": None,
            "gold_date": None,
            "gold_is_receipt": True,
        },
        {
            "receipt_ok": True,
            "total_ok": False,
            "date_ok": True,
            "ok": False,
            "hallucinated_total": True,
            "hallucinated_date": False,
            "gold_total": None,
            "gold_date": "2015-01-01",
            "gold_is_receipt": True,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["n"] == 2
    assert summary["date"] == "2/2"
    assert summary["hallucinated_total"]["count"] == 1
    assert summary["hallucinated_total"]["gold_null_total_cases"] == 2
    assert summary["hallucinated_total"]["rate"] == 0.5
