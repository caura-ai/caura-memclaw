"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import httpx
import openai

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_HTTPX_MAX_CONNECTIONS,
    OPENAI_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
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
        #
        # Explicit ``http_client`` with ``httpx.Limits`` sized for our
        # bulk-write fan-out (CAURA-627). Same rationale as
        # ``OpenAILLMProvider``: the SDK's default httpx pool (100 max
        # / 20 keepalive) saturates under storm load and queues other
        # tenants' embed calls at the pool layer.
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=OPENAI_HTTPX_MAX_CONNECTIONS,
                    max_keepalive_connections=OPENAI_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
                ),
            ),
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        """Close the underlying httpx pool cleanly.

        Without this, ``asyncio`` debug mode emits ``ResourceWarning:
        Unclosed <httpx.AsyncClient>`` when the provider is GC'd —
        noisy in tests and a leak in long-lived processes that rotate
        client instances. Idempotent; safe to call multiple times.
        """
        await self._client.close()

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
