from types import SimpleNamespace

from app.core.config import Settings
from app.domain.schemas import ReceiptLLMOutput
from app.llm.provider import OpenRouterProvider


class CapturingCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace()


def test_provider_requests_strict_receipt_schema():
    completions = CapturingCompletions()

    provider = OpenRouterProvider(
        Settings(openrouter_api_key="test-key")
    )

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    provider.complete(
        [{"role": "user", "content": "Extract this receipt"}]
    )

    response_format = completions.kwargs["response_format"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "receipt_extraction"
    assert response_format["json_schema"]["strict"] is True
    assert (
        response_format["json_schema"]["schema"]
        == ReceiptLLMOutput.model_json_schema()
    )
    assert (
    completions.kwargs["extra_body"]["provider"]["require_parameters"]
    is True
    )

