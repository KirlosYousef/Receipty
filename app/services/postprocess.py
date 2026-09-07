from app.domain.schemas import ReceiptExtract

CODES = {
    "egp": "EGP",
    "le": "EGP",
    "ج.م": "EGP",
    "usd": "USD",
    "$": "USD",
    "eur": "EUR",
    "€": "EUR",
}


def infer_currency(text: str) -> str | None:
    t = text.lower()
    hits = [code for needle, code in CODES.items() if needle in t]
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


def apply_postprocess(row: ReceiptExtract, currency_hint: str) -> ReceiptExtract:
    if not row.is_receipt:
        row.merchant = None
        row.total = None
        row.currency = None
        row.date = None
        row.tax = None
        row.needs_review = True
        return row
    if row.total is None:
        row.needs_review = True
    if row.currency is None:
        row.currency = infer_currency(currency_hint)
    return row
