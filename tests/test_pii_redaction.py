import json
import logging
from pathlib import Path

from app.observability.redact import (
    RedactingFilter,
    install_redacting_filter,
    redact_pii,
)
from app.observability.tracing import SpanRecorder


def test_redact_payment_cards():
    assert (
        redact_pii("Charged to card 4111 2222 3333 4444 on file")
        == "Charged to card [REDACTED_CARD] on file"
    )
    assert redact_pii("Card: 4111-2222-3333-4444") == "Card: [REDACTED_CARD]"
    assert (
        redact_pii("AMEX 3782 822463 10005 approved") == "AMEX [REDACTED_CARD] approved"
    )


def test_keep_dates_and_money_unaltered():
    receipt_line = "Taco Bell 2024-01-15 Total 1,234.56 USD Tax 12.50"
    assert redact_pii(receipt_line) == receipt_line


def test_redact_emails_and_phones():
    text = "Contact cashier at store54@fastfood.com or call (555) 234-5678 for refunds"
    scrubbed = redact_pii(text)
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "store54@fastfood.com" not in scrubbed
    assert "555" not in scrubbed


def test_redact_credentials_and_bearer_tokens():
    text = "Header Bearer eyJhbGciOiJIUzI1NiJ9.secret and key sk-proj_9876543210abcdef123456"
    scrubbed = redact_pii(text)
    assert "Bearer [REDACTED_KEY]" in scrubbed
    assert "[REDACTED_KEY]" in scrubbed
    assert "eyJhbGci" not in scrubbed
    assert "sk-proj" not in scrubbed


def test_redacting_filter_scrubs_logger_output():
    logger = logging.getLogger("test_pii_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    records: list[str] = []

    class CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    handler = CapturingHandler()
    logger.addHandler(handler)
    filter_instance = install_redacting_filter(logger)
    assert isinstance(filter_instance, RedactingFilter)

    logger.info("Payment card %s email %s", "4111 2222 3333 4444", "user@example.com")
    assert len(records) == 1
    assert "[REDACTED_CARD]" in records[0]
    assert "[REDACTED_EMAIL]" in records[0]
    assert "4111" not in records[0]
    assert "user@example.com" not in records[0]


def test_span_recorder_scrubs_pii_from_trace_attributes(tmp_path: Path):
    path = tmp_path / "traces.jsonl"
    recorder = SpanRecorder(path)
    with recorder.span(
        "customer_step",
        {"user_contact": "call (555) 123-4567 or email vip@acme.org"},
    ):
        pass

    line = json.loads(path.read_text().strip())
    assert line["name"] == "customer_step"
    assert "[REDACTED_PHONE]" in line["user_contact"]
    assert "[REDACTED_EMAIL]" in line["user_contact"]
    assert "vip@acme.org" not in line["user_contact"]


def test_redact_pii_empty_string():
    assert redact_pii("") == ""


def test_redacting_filter_handles_dict_args_and_exc_text():
    logger = logging.getLogger("test_dict_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    filter_instance = install_redacting_filter(logger)
    second_filter = install_redacting_filter(logger)
    assert filter_instance is second_filter

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Error in transaction: %(account)s code %(code)s",
        args={"account": "user@domain.com", "code": 404},
        exc_info=None,
    )
    record.exc_text = "Exception at key sk-proj_abcdef12345678901234"
    filter_instance.filter(record)
    assert record.args["account"] == "[REDACTED_EMAIL]"
    assert record.args["code"] == 404
    assert "[REDACTED_KEY]" in record.exc_text
