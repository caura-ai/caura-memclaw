"""Platform default embedding-provider singleton.

Pre-built at startup from ``PLATFORM_EMBEDDING_*`` env vars. Returned by
:func:`common.embedding.get_embedding_provider` when a tenant has no
credentials configured — tier 2 in the three-tier resolution:

    Tenant key  →  Platform singleton  →  FakeEmbeddingProvider

Security: keys are sealed into the singleton at construction time and
never enter tenant-configurable code paths.

Env-driven (no service-config dependency) so both core-api and
core-worker initialise the same singleton from the same env shape.
"""

from __future__ import annotations

import logging
import os

from common.embedding.constants import OPENAI_EMBEDDING_MODEL, VERTEX_EMBEDDING_MODEL
from common.embedding.protocols import EmbeddingProvider
from common.embedding.providers.openai import OpenAIEmbeddingProvider
from common.embedding.providers.vertex import VertexEmbeddingProvider
from common.provider_names import ProviderName

logger = logging.getLogger(__name__)

_platform_embedding: EmbeddingProvider | None = None
_platform_init_errors: list[str] = []


def init_platform_embedding() -> None:
    """Build the singleton from ``PLATFORM_EMBEDDING_*`` env vars.

    Idempotent — call once during service lifespan startup. Subsequent
    calls reset and rebuild the singleton (useful for tests).
    """
    global _platform_embedding
    _platform_embedding = None
    _platform_init_errors.clear()

    provider = os.environ.get("PLATFORM_EMBEDDING_PROVIDER", "")
    if not provider:
        # No platform default configured — tenants without their own
        # keys will fall through to FakeEmbeddingProvider.
        return

    if provider == ProviderName.OPENAI:
        api_key = os.environ.get("PLATFORM_EMBEDDING_API_KEY", "")
        if not api_key:
            logger.warning(
                "PLATFORM_EMBEDDING_PROVIDER=openai but no PLATFORM_EMBEDDING_API_KEY"
            )
            _platform_init_errors.append("openai-embedding-config")
            return
        try:
            embed_model = (
                os.environ.get("PLATFORM_EMBEDDING_MODEL") or OPENAI_EMBEDDING_MODEL
            )
            _platform_embedding = OpenAIEmbeddingProvider(
                api_key=api_key, model=embed_model
            )
            logger.info("Platform embedding: openai/%s", embed_model)
        except Exception:
            logger.exception("Failed to initialize platform OpenAI embedding provider")
            _platform_init_errors.append("openai-embedding")
        return

    if provider == ProviderName.VERTEX:
        # Allow falling back to PLATFORM_LLM_GCP_* for project + location.
        project_id = os.environ.get(
            "PLATFORM_EMBEDDING_GCP_PROJECT_ID"
        ) or os.environ.get("PLATFORM_LLM_GCP_PROJECT_ID", "")
        if not project_id:
            logger.warning(
                "PLATFORM_EMBEDDING_PROVIDER=vertex but no "
                "PLATFORM_EMBEDDING_GCP_PROJECT_ID or PLATFORM_LLM_GCP_PROJECT_ID"
            )
            _platform_init_errors.append("vertex-embedding-config")
            return
        try:
            location = (
                os.environ.get("PLATFORM_EMBEDDING_GCP_LOCATION")
                or os.environ.get("PLATFORM_LLM_GCP_LOCATION")
                or "us-central1"
            )
            embed_model = (
                os.environ.get("PLATFORM_EMBEDDING_MODEL") or VERTEX_EMBEDDING_MODEL
            )
            _platform_embedding = VertexEmbeddingProvider(
                project_id=project_id, location=location, model=embed_model
            )
            logger.info(
                "Platform embedding: vertex/%s (%s/%s)",
                embed_model,
                project_id,
                location,
            )
        except Exception:
            logger.exception("Failed to initialize platform Vertex embedding provider")
            _platform_init_errors.append("vertex-embedding")
        return

    logger.warning(
        "Unknown PLATFORM_EMBEDDING_PROVIDER=%r — no platform embedding will be configured",
        provider,
    )
    _platform_init_errors.append("unknown-embedding-provider")


def get_platform_embedding() -> EmbeddingProvider | None:
    """Return the platform embedding singleton, or ``None`` if unset."""
    return _platform_embedding


def get_platform_init_errors() -> list[str]:
    """Provider names that failed during the most recent ``init_platform_embedding`` call."""
    return list(_platform_init_errors)
