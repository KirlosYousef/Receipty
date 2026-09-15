from __future__ import annotations

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


def hallucinated_total(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    """Gold total is absent/unreadable, but the prediction invented one."""
    return gold.get("total") is None and pred.total is not None


def hallucinated_date(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    """Gold date is absent/unreadable, but the prediction invented one."""
    return gold.get("date") is None and pred.date is not None


def expected_outcome_from_gold(gold: dict[str, Any]) -> str:
    """Prefer an explicit label; otherwise apply the review policy to gold fields."""
    explicit = gold.get("expected_outcome")
    if explicit:
        return str(explicit)
    if not gold.get("is_receipt"):
        return Outcome.not_receipt.value
    if gold.get("total") is None or gold.get("currency") is None:
        return Outcome.needs_review.value
    return Outcome.success.value


def score_row(pred: ReceiptExtract, gold: dict[str, Any]) -> dict[str, Any]:
    r_ok = receipt_hit(pred, gold)
    t_ok = total_hit(pred, gold)
    d_ok = date_hit(pred, gold)
    expected = expected_outcome_from_gold(gold)
    pred_outcome = pred.outcome.value if pred.outcome is not None else None
    return {
        "receipt_ok": r_ok,
        "total_ok": t_ok,
        "date_ok": d_ok,
        "ok": r_ok and t_ok and d_ok,
        "hallucinated_total": hallucinated_total(pred, gold),
        "hallucinated_date": hallucinated_date(pred, gold),
        "pred_outcome": pred_outcome,
        "expected_outcome": expected,
        "outcome_ok": pred_outcome == expected if pred_outcome is not None else None,
        "pred_merchant": pred.merchant,
        "pred_total": None if pred.total is None else str(pred.total),
        "pred_date": None if pred.date is None else str(pred.date),
        "gold_merchant": gold.get("merchant"),
        "gold_total": gold_total_value(gold),
        "gold_date": gold.get("date"),
        "gold_is_receipt": bool(gold.get("is_receipt")),
    }


def gold_total_value(gold: dict[str, Any]) -> float | None:
    t = gold.get("total")
    return None if t is None else float(t)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if "receipt_ok" in r]
    n = len(scored)
    errors = sum(1 for r in rows if r.get("error"))
    if n == 0:
        return {
            "n": 0,
            "errors": errors,
            "is_receipt": "0/0",
            "total": "0/0",
            "date": "0/0",
            "ok": "0/0",
            "is_receipt_correct": 0,
            "total_correct": 0,
            "date_correct": 0,
            "ok_correct": 0,
            "hallucinated_total": {
                "count": 0,
                "gold_null_total_cases": 0,
                "rate": None,
            },
            "hallucinated_date": {
                "count": 0,
                "gold_null_date_cases": 0,
                "rate": None,
            },
            "needs_review": {
                "precision": None,
                "recall": None,
                "pred_count": 0,
                "gold_count": 0,
                "true_positives": 0,
            },
        }

    receipt_correct = sum(1 for r in scored if r["receipt_ok"])
    total_correct = sum(1 for r in scored if r["total_ok"])
    date_correct = sum(1 for r in scored if r["date_ok"])
    ok_correct = sum(1 for r in scored if r["ok"])

    # Keep the historical runner printout: receipt-positive hits only.
    receipt_positive_hits = sum(
        1
        for r in scored
        if r["receipt_ok"] and r.get("pred_outcome") != Outcome.not_receipt.value
    )

    gold_null_totals = [r for r in scored if r.get("gold_total") is None]
    gold_null_dates = [r for r in scored if r.get("gold_date") is None]
    hall_totals = sum(1 for r in scored if r.get("hallucinated_total"))
    hall_dates = sum(1 for r in scored if r.get("hallucinated_date"))

    pred_review = [
        r for r in scored if r.get("pred_outcome") == Outcome.needs_review.value
    ]
    gold_review = [
        r for r in scored if r.get("expected_outcome") == Outcome.needs_review.value
    ]
    true_positive_review = sum(
        1
        for r in scored
        if r.get("pred_outcome") == Outcome.needs_review.value
        and r.get("expected_outcome") == Outcome.needs_review.value
    )

    return {
        "n": n,
        "errors": errors,
        "is_receipt": f"{receipt_positive_hits}/{n}",
        "total": f"{total_correct}/{n}",
        "date": f"{date_correct}/{n}",
        "ok": f"{ok_correct}/{n}",
        "is_receipt_correct": receipt_positive_hits,
        "receipt_ok_correct": receipt_correct,
        "total_correct": total_correct,
        "date_correct": date_correct,
        "ok_correct": ok_correct,
        "hallucinated_total": {
            "count": hall_totals,
            "gold_null_total_cases": len(gold_null_totals),
            "rate": _ratio(hall_totals, len(gold_null_totals)),
        },
        "hallucinated_date": {
            "count": hall_dates,
            "gold_null_date_cases": len(gold_null_dates),
            "rate": _ratio(hall_dates, len(gold_null_dates)),
        },
        "needs_review": {
            "precision": _ratio(true_positive_review, len(pred_review)),
            "recall": _ratio(true_positive_review, len(gold_review)),
            "pred_count": len(pred_review),
            "gold_count": len(gold_review),
            "true_positives": true_positive_review,
        },
    }


def _ratio(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den
