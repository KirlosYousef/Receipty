from __future__ import annotations

import logging
import re
from typing import Any

# Card PANs: 13 to 19 digits with optional spaces or dashes
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# Email addresses
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Standard phone numbers (US and international forms with 10+ digits)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Secret keys / Bearer tokens
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.~+/]+=*", re.IGNORECASE)
_API_KEY_RE = re.compile(r"\b(?:sk|pk)(?:-[a-zA-Z0-9]+)?_[a-zA-Z0-9_\-]{16,}\b")


def _redact_cards(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = [c for c in raw if c.isdigit()]
        if 13 <= len(digits) <= 19:
            return "[REDACTED_CARD]"
        return raw

    return _CARD_RE.sub(_repl, text)


def redact_pii(text: str) -> str:
    """Scrub payment card numbers, emails, phone numbers, and credentials."""
    if not text:
        return text
    scrubbed = _redact_cards(text)
    scrubbed = _EMAIL_RE.sub("[REDACTED_EMAIL]", scrubbed)
    scrubbed = _PHONE_RE.sub("[REDACTED_PHONE]", scrubbed)
    scrubbed = _BEARER_RE.sub("Bearer [REDACTED_KEY]", scrubbed)
    return _API_KEY_RE.sub("[REDACTED_KEY]", scrubbed)


class RedactingFilter(logging.Filter):
    """Scrub PII from log messages and arguments before writing."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(self._clean(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self._clean(v) for k, v in record.args.items()}
        if isinstance(record.exc_text, str):
            record.exc_text = redact_pii(record.exc_text)
        return True

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return redact_pii(value)
        return value


def install_redacting_filter(
    target_logger: logging.Logger | None = None,
) -> RedactingFilter:
    log_obj = target_logger or logging.getLogger()
    for existing in log_obj.filters:
        if isinstance(existing, RedactingFilter):
            return existing
    filter_instance = RedactingFilter()
    log_obj.addFilter(filter_instance)
    return filter_instance
