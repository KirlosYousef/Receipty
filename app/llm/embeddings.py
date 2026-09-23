from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.core.config import Settings
from app.observability.tracing import SpanRecorder
from app.repository.receipts import EMBEDDING_DIMENSIONS


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic embedding for tests. Not a semantic model."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        seed = digest
        while len(values) < EMBEDDING_DIMENSIONS:
            for byte in seed:
                values.append((byte / 127.5) - 1.0)
                if len(values) == EMBEDDING_DIMENSIONS:
                    break
            seed = hashlib.sha256(seed).digest()
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class OpenRouterEmbeddings:
    def __init__(self, settings: Settings, *, spans: SpanRecorder | None = None):
        from openai import OpenAI

        from app.core.exceptions import ProviderError

        if not settings.openrouter_api_key:
            raise ProviderError("OPENROUTER_API_KEY is not set")
        self._model = settings.embedding_model
        self._spans = spans or SpanRecorder(None)
        self._client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            max_retries=0,
        )

    def embed(self, text: str) -> list[float]:
        with self._spans.span(
            f"embeddings {self._model}",
            {
                "gen_ai.operation.name": "embeddings",
                "gen_ai.provider.name": "openrouter",
                "gen_ai.request.model": self._model,
            },
        ):
            response = self._client.embeddings.create(model=self._model, input=text)
        vector = list(response.data[0].embedding)
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"embedding dimension {len(vector)} != {EMBEDDING_DIMENSIONS}"
            )
        return vector
