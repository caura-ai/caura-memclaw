"""OpenAI-compatible embedding provider."""

from __future__ import annotations

import math

import openai

from common.constants import VECTOR_DIM
from common.embedding.constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
)


class OpenAIEmbeddingProvider:
    """Embedding provider using the OpenAI embeddings API.

    Also serves any OpenAI-compatible endpoint (TEI, vLLM, etc.) by setting
    ``base_url``. Supports asymmetric query/doc encoding for instruction-aware
    embedders (Qwen3-Embedding, e5-instruct, ...) via :meth:`embed_query` and
    the ``query_instruction`` constructor arg. Supports Matryoshka-style
    output truncation via ``truncate_to_dim`` so models with native dim
    larger than the pgvector column can be used without a schema migration.
    """

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_EMBEDDING_MODEL,
        base_url: str | None = None,
        send_dimensions: bool = True,
        query_instruction: str | None = None,
        truncate_to_dim: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._send_dimensions = send_dimensions
        self._query_instruction = query_instruction
        self._truncate_to_dim = truncate_to_dim
        # Explicit per-call timeout — without this the SDK rides
        # httpx's default 600s read timeout, and a single hung
        # api.openai.com response would silently eat the worker's
        # entire ack budget. Mirrors the same env-driven default that
        # gates ``OpenAILLMProvider``.
        client_kwargs: dict = {
            "api_key": api_key,
            "timeout": OPENAI_REQUEST_TIMEOUT_SECONDS,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def _postprocess(self, emb: list[float]) -> list[float]:
        # Matryoshka truncation: slice to ``truncate_to_dim`` and L2-renormalize.
        # Models trained with MRL (Qwen3-Embedding, jina-v3, snowflake-arctic-l)
        # produce vectors that stay coherent under truncation, but the resulting
        # vector is no longer unit-norm — we re-normalize so cosine similarity
        # at the pgvector layer stays correct.
        if self._truncate_to_dim and len(emb) > self._truncate_to_dim:
            emb = list(emb[: self._truncate_to_dim])
            n = math.sqrt(sum(x * x for x in emb))
            if n > 0:
                emb = [x / n for x in emb]
        return emb

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text (document/ingest path)."""
        kwargs: dict = {"model": self._model, "input": text}
        if self._send_dimensions:
            kwargs["dimensions"] = VECTOR_DIM
        response = await self._client.embeddings.create(**kwargs)
        return self._postprocess(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call."""
        kwargs: dict = {"model": self._model, "input": texts}
        if self._send_dimensions:
            kwargs["dimensions"] = VECTOR_DIM
        response = await self._client.embeddings.create(**kwargs)
        # OpenAI returns embeddings sorted by index
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [self._postprocess(item.embedding) for item in sorted_data]

    async def embed_query(
        self, text: str, instruction: str | None = None
    ) -> list[float]:
        """Generate an embedding vector for a search-side query.

        For instruction-aware models (Qwen3-Embedding, e5-instruct), prepends
        the resolved instruction in the convention these models were trained
        with: ``"Instruct: <task>\\nQuery: <text>"``. The instruction is
        resolved as: per-call *instruction* arg → constructor
        *query_instruction* → no prefix.

        For symmetric models (bge-m3, snowflake-arctic-l, gte-en-v1.5,
        OpenAI text-embedding-3-small), pass no instruction at construction
        time and this is equivalent to :meth:`embed`.
        """
        instr = instruction or self._query_instruction
        if instr:
            text = f"Instruct: {instr}\nQuery: {text}"
        return await self.embed(text)
