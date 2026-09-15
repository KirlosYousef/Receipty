EXTRACTION_PROMPTS = {
    "extraction-v1": """Extract a purchase receipt.
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
""",
    "extraction-v2-evidence": """Extract a purchase receipt from the input.
Not a receipt → is_receipt=false, other fields null.
Return these fields: is_receipt, merchant, total, currency, date, and tax.
Copy values only when they are visibly supported by the receipt. Never infer a missing,
ambiguous, or unreadable value. For total, choose only a clearly labeled final amount
such as TOTAL, AMOUNT DUE, or BALANCE; otherwise return null. Preserve a visible
transaction date as YYYY-MM-DD, or return null when it is absent or ambiguous.

Reply with JSON only, no markdown.
""",
}

DEFAULT_PROMPT_VERSION = "extraction-v1"
EXTRACTION_PROMPT = EXTRACTION_PROMPTS[DEFAULT_PROMPT_VERSION]


def prompt_for(version: str) -> str:
    try:
        return EXTRACTION_PROMPTS[version]
    except KeyError as exc:
        raise ValueError(f"unknown prompt version: {version}") from exc
