"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import openai

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
)


class OpenAIEmbeddingProvider:
    """Embedding provider using the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_EMBEDDING_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # Explicit per-call timeout — without this the SDK rides
        # httpx's default 600s read timeout, and a single hung
        # api.openai.com response would silently eat the worker's
        # entire ack budget. Mirrors the same env-driven default that
        # gates ``OpenAILLMProvider``.
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=VECTOR_DIM,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=VECTOR_DIM,
        )
        # OpenAI returns embeddings sorted by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
