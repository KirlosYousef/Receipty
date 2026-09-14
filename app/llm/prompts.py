EXTRACTION_PROMPT = """Extract a purchase receipt.
Not a receipt → is_receipt=false, other fields null.
Return these fields: is_receipt, merchant, total, currency, date, and tax.
Copy merchant, total, date, and tax exactly when they are visible. Never invent a value.
Unreadable or missing total, date, or tax → set that field to null.
Extract the transaction date whenever it is visibly readable, including numeric
forms such as 5/26/2016, 05/26/2016, 2016-05-26, 06Aug'16, and dates with month names.

Return it as YYYY-MM-DD.
Do not infer or fabricate a date. If it is absent, ambiguous, or unreadable,
return null.

Reply with JSON only, no markdown.
"""
