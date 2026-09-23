from types import SimpleNamespace

from app.domain.schemas import Outcome
from app.services.extraction import ExtractionService

NEEDS_REVIEW = """
{
  "is_receipt": true,
  "merchant": "Cafe",
  "total": null,
  "currency": "USD",
  "date": "2024-01-15",
  "tax": null
}
"""

SUCCESS = """
{
  "is_receipt": true,
  "merchant": "Cafe",
  "total": "12.50",
  "currency": "USD",
  "date": "2024-01-15",
  "tax": "1.00"
}
"""

NOT_RECEIPT = """
{
  "is_receipt": false,
  "merchant": null,
  "total": null,
  "currency": null,
  "date": null,
  "tax": null
}
"""


class CountingProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, messages, *, request_id=None):
        del messages, request_id
        self.calls += 1
        return SimpleNamespace(
            model="fake",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1, cost=0, model_extra={}
            ),
        )


def test_needs_review_is_sent_once_to_the_stronger_model():
    cheap = CountingProvider(NEEDS_REVIEW)
    strong = CountingProvider(SUCCESS)
    logged: list[str] = []

    class Logger:
        def log(self, completion, kind: str) -> None:
            del completion
            logged.append(kind)

    row = ExtractionService(cheap, Logger(), escalation=strong).extract_from_text(
        "Cafe"
    )
    assert row.outcome == Outcome.success
    assert row.total == 12.50
    assert cheap.calls == 1
    assert strong.calls == 1
    assert logged == ["text", "text_escalation"]


def test_bad_json_is_sent_once_to_the_stronger_model():
    cheap = CountingProvider("this is not json")
    strong = CountingProvider(SUCCESS)
    row = ExtractionService(cheap, escalation=strong).extract_from_text("Cafe")
    assert row.outcome == Outcome.success
    assert cheap.calls == 1
    assert strong.calls == 1


def test_success_stays_on_the_first_model():
    cheap = CountingProvider(SUCCESS)
    strong = CountingProvider(NEEDS_REVIEW)
    row = ExtractionService(cheap, escalation=strong).extract_from_text("Cafe")
    assert row.outcome == Outcome.success
    assert cheap.calls == 1
    assert strong.calls == 0


def test_not_receipt_stays_on_the_first_model():
    cheap = CountingProvider(NOT_RECEIPT)
    strong = CountingProvider(SUCCESS)
    row = ExtractionService(cheap, escalation=strong).extract_from_text("hello")
    assert row.outcome == Outcome.not_receipt
    assert strong.calls == 0
