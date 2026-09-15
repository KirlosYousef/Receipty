from __future__ import annotations

import math
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


def merchant_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool | None:
    """Compare receipt merchant names when the gold label supplies one."""
    gold_merchant = gold.get("merchant")
    if not gold.get("is_receipt") or gold_merchant is None:
        return None
    return _normalize_text(pred.merchant) == _normalize_text(gold_merchant)


def currency_hit(pred: ReceiptExtract, gold: dict[str, Any]) -> bool | None:
    """Compare receipt currency only when it was explicitly labelled."""
    gold_currency = gold.get("currency")
    if not gold.get("is_receipt") or gold_currency is None:
        return None
    return pred.currency == gold_currency


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split()).casefold()


def hallucinated_total(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    """Gold total is absent/unreadable, but the prediction invented one."""
    return (
        bool(gold.get("is_receipt"))
        and gold.get("total") is None
        and pred.total is not None
    )


def hallucinated_date(pred: ReceiptExtract, gold: dict[str, Any]) -> bool:
    """Gold date is absent/unreadable, but the prediction invented one."""
    return (
        bool(gold.get("is_receipt"))
        and gold.get("date") is None
        and pred.date is not None
    )


def score_row(pred: ReceiptExtract, gold: dict[str, Any]) -> dict[str, Any]:
    r_ok = receipt_hit(pred, gold)
    t_ok = total_hit(pred, gold)
    d_ok = date_hit(pred, gold)
    m_ok = merchant_hit(pred, gold)
    c_ok = currency_hit(pred, gold)
    pred_outcome = pred.outcome.value if pred.outcome is not None else None
    return {
        "receipt_ok": r_ok,
        "total_ok": t_ok,
        "date_ok": d_ok,
        "merchant_ok": m_ok,
        "currency_ok": c_ok,
        "ok": r_ok and t_ok and d_ok,
        "hallucinated_total": hallucinated_total(pred, gold),
        "hallucinated_date": hallucinated_date(pred, gold),
        "pred_outcome": pred_outcome,
        "pred_merchant": pred.merchant,
        "pred_total": None if pred.total is None else str(pred.total),
        "pred_currency": pred.currency,
        "pred_date": None if pred.date is None else str(pred.date),
        "pred_tax": None if pred.tax is None else str(pred.tax),
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
            "latency_ms": _latency_summary([]),
            "tokens": _token_summary([]),
            "cost_usd": _cost_summary([]),
            "class_conditional": _class_conditional_summary([]),
            "fields": _field_summaries([]),
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

    gold_null_totals = [
        r for r in scored if r.get("gold_is_receipt") and r.get("gold_total") is None
    ]
    gold_null_dates = [
        r for r in scored if r.get("gold_is_receipt") and r.get("gold_date") is None
    ]
    hall_totals = sum(1 for r in scored if r.get("hallucinated_total"))
    hall_dates = sum(1 for r in scored if r.get("hallucinated_date"))
    latencies = [
        float(row["latency_ms"]) for row in scored if row.get("latency_ms") is not None
    ]
    usage_rows = [
        row
        for row in scored
        if row.get("prompt_tokens") is not None
        and row.get("completion_tokens") is not None
        and row.get("total_tokens") is not None
    ]
    costs = [float(row["usd"]) for row in scored if row.get("usd") is not None]

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
        "latency_ms": _latency_summary(latencies),
        "tokens": _token_summary(usage_rows),
        "cost_usd": _cost_summary(costs),
        "class_conditional": _class_conditional_summary(scored),
        "fields": _field_summaries(scored),
    }


def _ratio(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "mean": None}
    return {
        "count": len(values),
        "p50": _nearest_rank_percentile(values, 0.50),
        "p95": _nearest_rank_percentile(values, 0.95),
        "mean": sum(values) / len(values),
    }


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def _token_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {
            "count": 0,
            "prompt_sum": None,
            "completion_sum": None,
            "total_sum": None,
            "mean_total": None,
        }
    prompt_sum = sum(int(row["prompt_tokens"]) for row in rows)
    completion_sum = sum(int(row["completion_tokens"]) for row in rows)
    total_sum = sum(int(row["total_tokens"]) for row in rows)
    return {
        "count": len(rows),
        "prompt_sum": prompt_sum,
        "completion_sum": completion_sum,
        "total_sum": total_sum,
        "mean_total": total_sum / len(rows),
    }


def _cost_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum": None, "mean": None}
    total = sum(values)
    return {
        "count": len(values),
        "sum": round(total, 6),
        "mean": round(total / len(values), 6),
    }


def _class_conditional_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "receipt": _class_summary(
            [row for row in rows if row.get("gold_is_receipt") is True]
        ),
        "non_receipt": _class_summary(
            [row for row in rows if row.get("gold_is_receipt") is False]
        ),
    }


def _class_summary(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    return {
        "n": len(rows),
        "receipt_accuracy": _accuracy(rows, "receipt_ok"),
        "total_accuracy": _accuracy(rows, "total_ok"),
        "date_accuracy": _accuracy(rows, "date_ok"),
        "combined_accuracy": _accuracy(rows, "ok"),
    }


def _field_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "merchant": _field_summary(rows, "merchant_ok"),
        "currency": _field_summary(rows, "currency_ok"),
        "tax": {
            "correct": None,
            "scored": 0,
            "accuracy": None,
            "reason": "labels do not include tax",
        },
    }


def _field_summary(
    rows: list[dict[str, Any]], key: str
) -> dict[str, int | float | None]:
    scored = [row for row in rows if row.get(key) is not None]
    correct = sum(1 for row in scored if row[key])
    return {
        "correct": correct,
        "scored": len(scored),
        "accuracy": _ratio(correct, len(scored)),
    }


def _accuracy(rows: list[dict[str, Any]], key: str) -> float | None:
    return _ratio(sum(1 for row in rows if row.get(key)), len(rows))
