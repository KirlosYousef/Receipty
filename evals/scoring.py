from decimal import Decimal
from typing import Any

from app.domain.schemas import ReceiptExtract


def receipt_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    return pred.is_receipt == bool(gold["is_receipt"])


def total_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    gold_total = gold.get("total")
    if gold_total is None and pred.total is None:
        return True
    if pred.total is None or gold_total is None:
        return False
    return float(pred.total) == float(gold_total)


def score_row(pred: ReceiptExtract, gold: dict[str, Any]) -> dict[str, Any]:
    r_ok = receipt_hit(pred, gold)
    t_ok = total_hit(pred, gold)
    return {
        "receipt_ok": r_ok,
        "total_ok": t_ok,
        "ok": r_ok and t_ok,
        "pred_merchant": pred.merchant,
        "pred_total": str(pred.total) if isinstance(pred.total, Decimal) else pred.total,
        "gold_merchant": gold.get("merchant"),
        "gold_total": gold_total_value(gold),
    }


def gold_total_value(gold: dict[str, Any]) -> float | None:
    t = gold.get("total")
    return None if t is None else float(t)
