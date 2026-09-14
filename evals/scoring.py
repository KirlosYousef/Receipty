from decimal import Decimal
from typing import Any

from app.domain.schemas import Outcome, ReceiptExtract


def receipt_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    if pred.outcome == Outcome.extraction_failed:
        return False  # the system made no classification; it cannot be correct
    return (pred.outcome != Outcome.not_receipt) == bool(gold["is_receipt"])


def total_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    gold_total = gold.get("total")
    if gold_total is None and pred.total is None:
        return True
    if pred.total is None or gold_total is None:
        return False
    return float(pred.total) == float(gold_total)


def date_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    gold_date = gold.get("date")
    if gold_date is None and pred.date is None:
        return True
    if pred.date is None or gold_date is None:
        return False
    return str(pred.date) == str(gold_date)


def score_row(pred: ReceiptExtract, gold: dict[str, Any]) -> dict[str, Any]:
    r_ok = receipt_hit(pred, gold)
    t_ok = total_hit(pred, gold)
    d_ok = date_hit(pred, gold)
    return {
        "receipt_ok": r_ok,
        "total_ok": t_ok,
        "date_ok": d_ok,
        "ok": r_ok and t_ok and d_ok,
        "pred_merchant": pred.merchant,
        "pred_total": str(pred.total),
        "pred_date": str(pred.date) if isinstance(pred.total, Decimal) else pred.total,
        "gold_merchant": gold.get("merchant"),
        "gold_total": gold_total_value(gold),
        "gold_date": gold.get("date"),
    }


def gold_total_value(gold: dict[str, Any]) -> float | None:
    t = gold.get("total")
    return None if t is None else float(t)
