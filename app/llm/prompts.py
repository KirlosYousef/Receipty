EXTRACTION_PROMPT = """Extract a purchase receipt.
Not a receipt → is_receipt=false, other fields null.
Return these fields: is_receipt, merchant, total, currency, date, tax, and needs_review.
Copy merchant, total, date, and tax exactly when they are visible. Never invent a value.
Unreadable or missing total, date, or tax → set that field to null.
Reply with JSON only, no markdown.
"""
