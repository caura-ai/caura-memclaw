"""Service-level embedding entrypoints with retry + degraded-state stats.

Reads provider selection from env (``EMBEDDING_PROVIDER``) when no
tenant override is supplied — both core-api and core-worker drive the
same code path, just with / without ``tenant_config``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from common.embedding._registry import get_embedding_provider
from common.embedding.constants import EMBEDDING_RETRY_ATTEMPTS, EMBEDDING_RETRY_DELAY_S

logger = logging.getLogger(__name__)


class _EmbeddingStats:
    """Track failure rate to surface a degraded-provider signal in logs.

    Three consecutive failures fire a single ERROR log (per cycle) so a
    sustained provider outage shows up loudly without spamming once-per-
    request. Reset on the next success.
    """

    def __init__(self) -> None:
        self.failures = 0
        self.successes = 0
        self.last_failure_time = 0.0
        self.consecutive_failures = 0
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self.successes += 1
            self.consecutive_failures = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_failure_time = time.monotonic()
            # Fire on first detection (failure 3) and then every 10
            # failures thereafter (13, 23, 33, …). Loud enough to alert
            # on a fresh outage, quiet enough not to spam during a
            # sustained provider degradation. The next success resets
            # the counter, so a subsequent streak re-fires from 3.
            if (
                self.consecutive_failures >= 3
                and (self.consecutive_failures - 3) % 10 == 0
            ):
                logger.error(
                    "Embedding service degraded: %d consecutive failures (total: %d/%d)",
                    self.consecutive_failures,
                    self.failures,
                    self.failures + self.successes,
                )


_stats = _EmbeddingStats()


def _resolve_provider_name(tenant_config: object | None) -> str:
    """Tenant override first, else ``EMBEDDING_PROVIDER`` env, else ``"fake"``."""
    if tenant_config is not None:
        name = getattr(tenant_config, "embedding_provider", None)
        if name:
            return name
    return os.environ.get("EMBEDDING_PROVIDER", "fake")


async def get_embeddings_batch(
    texts: list[str], tenant_config: object | None = None
) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single API call.

    Raises any provider-side exception. ``ValueError`` is treated as a
    programming error and re-raised without affecting failure stats.
    """
    provider_name = _resolve_provider_name(tenant_config)
    provider = get_embedding_provider(provider_name, tenant_config)

    try:
        result = await provider.embed_batch(texts)
        await _stats.record_success()
        return result
    except ValueError:
        raise
    except Exception:
        await _stats.record_failure()
        raise


async def get_embedding(
    text: str, tenant_config: object | None = None
) -> list[float] | None:
    """Generate an embedding with retry. Returns ``None`` on exhausted retries.

    Caller-friendly degradation: a transient OpenAI/Vertex hiccup
    returns ``None`` rather than raising, so write paths can persist
    rows with ``embedding=NULL`` and let the async-embed worker backfill.

    This is the document/ingest path. For search-side queries that should
    pass through an instruction-aware encoder (Qwen3-Embedding, e5-instruct,
    etc.), use :func:`get_query_embedding` instead.
    """
    provider_name = _resolve_provider_name(tenant_config)
    provider = get_embedding_provider(provider_name, tenant_config)

    for attempt in range(1, EMBEDDING_RETRY_ATTEMPTS + 1):
        try:
            result = await provider.embed(text)
            await _stats.record_success()
            return result
        except (
            Exception
        ):  # Intentionally broad: must catch all provider-specific errors during retry
            logger.warning(
                "Embedding attempt %d/%d failed",
                attempt,
                EMBEDDING_RETRY_ATTEMPTS,
                exc_info=True,
            )
            if attempt < EMBEDDING_RETRY_ATTEMPTS:
                await asyncio.sleep(EMBEDDING_RETRY_DELAY_S * attempt)
    await _stats.record_failure()
    logger.error(
        "Embedding failed after %d attempts, returning None", EMBEDDING_RETRY_ATTEMPTS
    )
    return None


async def get_query_embedding(
    text: str,
    tenant_config: object | None = None,
    instruction: str | None = None,
) -> list[float] | None:
    """Generate a query-side embedding (instruction-aware, asymmetric).

    For instruction-aware models (Qwen3-Embedding, e5-instruct, KaLM), the
    query encoder expects a task-description prefix that documents do not
    receive. Providers that don't have an instruction-aware mode (OpenAI
    text-embedding-3-small, bge-base, snowflake-m, gte-en-v1.5) silently
    fall back to the symmetric :meth:`embed` path — same behavior as
    :func:`get_embedding`.

    Same retry / degradation semantics as :func:`get_embedding`: returns
    ``None`` after exhausted retries rather than raising, so search routes
    can degrade gracefully (typically by raising 503 to the caller).
    """
    provider_name = _resolve_provider_name(tenant_config)
    provider = get_embedding_provider(provider_name, tenant_config)

    # Backward-compat: providers that don't yet implement embed_query
    # (Vertex, Fake, Local) auto-fall-back via the protocol's default
    # behavior. ``hasattr`` guard is belt-and-braces: we accept any
    # provider that exposes the method, even if the type-system view of
    # the Protocol is broader.
    embed_call = (
        provider.embed_query(text, instruction)
        if hasattr(provider, "embed_query")
        else provider.embed(text)
    )

    for attempt in range(1, EMBEDDING_RETRY_ATTEMPTS + 1):
        try:
            # Re-build the coroutine on retry — coroutines aren't reusable.
            if attempt > 1:
                embed_call = (
                    provider.embed_query(text, instruction)
                    if hasattr(provider, "embed_query")
                    else provider.embed(text)
                )
            result = await embed_call
            await _stats.record_success()
            return result
        except Exception:
            logger.warning(
                "Query embedding attempt %d/%d failed",
                attempt,
                EMBEDDING_RETRY_ATTEMPTS,
                exc_info=True,
            )
            if attempt < EMBEDDING_RETRY_ATTEMPTS:
                await asyncio.sleep(EMBEDDING_RETRY_DELAY_S * attempt)
    await _stats.record_failure()
    logger.error(
        "Query embedding failed after %d attempts, returning None",
        EMBEDDING_RETRY_ATTEMPTS,
    )
    return None
