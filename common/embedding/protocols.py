"""Embedding-provider protocol shared across services."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Async embedding provider for single and batch text embedding."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        The returned vector size is determined by the provider and model
        configuration. Callers should not assume a specific dimensionality
        — the system-wide ``VECTOR_DIM`` constant and the pgvector column
        definition enforce consistency at the storage layer.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call.

        Same dimensionality contract as :meth:`embed`.
        """
        ...
