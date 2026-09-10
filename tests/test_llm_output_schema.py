from app.domain.schemas import ReceiptLLMOutput


def test_llm_output_schema_contains_only_extracted_fields():
    schema = ReceiptLLMOutput.model_json_schema()

    assert set(schema["properties"]) == {
        "is_receipt",
        "merchant",
        "total",
        "currency",
        "date",
        "tax",
    }

    assert set(schema["required"]) == {
        "is_receipt",
        "merchant",
        "total",
        "currency",
        "date",
        "tax",
    }

    assert schema["additionalProperties"] is False
    assert "outcome" not in schema["properties"]
