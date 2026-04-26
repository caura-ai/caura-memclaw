"""Vertex AI embedding provider.

The Vertex SDK is synchronous — every call is wrapped in
``asyncio.to_thread()`` to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from common.embedding.constants import VERTEX_EMBEDDING_MODEL


class VertexEmbeddingProvider:
    """Embedding provider using Vertex AI ``TextEmbeddingModel``."""

    def __init__(
        self,
        project_id: str,
        location: str,
        model: str = VERTEX_EMBEDDING_MODEL,
    ) -> None:
        self._project_id = project_id
        self._location = location
        self._model = model
        # Lazy-initialised + cached. ``aiplatform.init`` +
        # ``from_pretrained`` is a network round trip — paying it once
        # per (project, location, model) per process. Double-checked
        # locking under ``_init_lock`` because ``_get_model`` runs
        # inside ``asyncio.to_thread`` (= a real thread pool), and two
        # concurrent first calls could otherwise both observe ``None``
        # and load the model twice.
        self._model_instance: Any = None
        self._init_lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "vertex"

    @property
    def model(self) -> str:
        return self._model

    def _get_model(self) -> Any:
        # Fast path: already initialised, no lock needed.
        if self._model_instance is not None:
            return self._model_instance
        # Slow path: take the lock, re-check (another thread may have
        # initialised while we waited), then load.
        with self._init_lock:
            if self._model_instance is None:
                from google.cloud import aiplatform
                from vertexai.language_models import TextEmbeddingModel

                aiplatform.init(project=self._project_id, location=self._location)
                self._model_instance = TextEmbeddingModel.from_pretrained(self._model)
        return self._model_instance

    def _embed_sync(self, text: str) -> list[float]:
        """Synchronous single-text embedding."""
        embeddings = self._get_model().get_embeddings([text])
        return embeddings[0].values

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch embedding."""
        embeddings = self._get_model().get_embeddings(texts)
        return [e.values for e in embeddings]

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text."""
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one call."""
        return await asyncio.to_thread(self._embed_batch_sync, texts)
