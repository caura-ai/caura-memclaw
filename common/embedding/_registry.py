"""Embedding-provider factory.

Resolves a concrete provider instance from the canonical name +
optional tenant config. Tier order:

    Tenant key  →  Platform singleton  →  FakeEmbeddingProvider

Env-driven (no service-config dependency) so both core-api (tenant-aware)
and core-worker (platform-only) can use it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict

from common.embedding.constants import OPENAI_EMBEDDING_MODEL
from common.embedding.protocols import EmbeddingProvider
from common.embedding.providers.fake import FakeEmbeddingProvider
from common.embedding.providers.local import LocalEmbedding
from common.embedding.providers.openai import OpenAIEmbeddingProvider
from common.provider_names import ProviderName

logger = logging.getLogger(__name__)

# Cache OpenAI provider instances by (api_key, model). Each instance
# holds a long-lived ``AsyncOpenAI`` (and therefore a long-lived
# ``httpx.AsyncClient`` connection pool); without the cache, every
# ``get_embedding`` call from core-api would build a fresh pool and
# pay a TLS handshake to api.openai.com on every request. Multi-tenant
# safe: keying on ``api_key`` keeps tenant-A's client from being
# reused for tenant-B's request. The platform-tier singleton has its
# own caching (managed in ``_platform.py``) — we don't double-cache it.
#
# LRU-bounded so a rotated/revoked tenant key eventually gets evicted
# instead of pinning a 401-returning client forever. ``OrderedDict``
# semantics: ``move_to_end`` on hit, ``popitem(last=False)`` on miss
# when full. 256 is well above any realistic tenant count for a single
# process; raise if that proves wrong.
#
# Eviction must explicitly close the evicted provider's httpx pool
# (CAURA-627). The OpenAI SDK now gets a user-provided ``http_client``
# (set in ``OpenAIEmbeddingProvider.__init__`` so we control pool
# sizing); per the SDK contract, user-provided clients are NOT closed
# on SDK teardown — caller owns cleanup. ``_get_or_create_openai_provider``
# below schedules ``aclose()`` on the evicted instance via
# ``asyncio.create_task`` so the pool drains in the background instead
# of leaking ``ResourceWarning`` and held connections under long-lived
# processes that rotate keys past the cache cap.
_OPENAI_CACHE_MAX = 256
_openai_provider_cache: OrderedDict[tuple[str, str], OpenAIEmbeddingProvider] = (
    OrderedDict()
)

# Strong references to in-flight ``aclose()`` cleanup tasks so the GC
# doesn't reap them mid-execution. ``asyncio.create_task`` returns a
# task that's kept alive only by user references; without this set the
# task could be collected before its coroutine finishes draining the
# httpx pool. The ``add_done_callback`` removes the task from the set
# once it completes so this doesn't grow unboundedly.
_background_tasks: set[asyncio.Task[None]] = set()


def _get_or_create_openai_provider(api_key: str, model: str) -> OpenAIEmbeddingProvider:
    """LRU-bounded ``OpenAIEmbeddingProvider`` lookup keyed on (api_key, model).

    Not strictly async-safe across concurrent cache misses for the same
    key — two coroutines can both observe a miss before either inserts.
    The race is harmless: the later insert overwrites the earlier with
    a functionally identical client (same api_key + model, no per-call
    state), and the earlier instance gets dropped on the next GC. Not
    worth an asyncio.Lock for the cost of a duplicated TLS handshake
    on the first concurrent miss.
    """
    cache_key = (api_key, model)
    cached = _openai_provider_cache.get(cache_key)
    if cached is not None:
        _openai_provider_cache.move_to_end(cache_key)
        return cached
    provider = OpenAIEmbeddingProvider(api_key=api_key, model=model)
    _openai_provider_cache[cache_key] = provider
    if len(_openai_provider_cache) > _OPENAI_CACHE_MAX:
        _, evicted = _openai_provider_cache.popitem(last=False)
        # Schedule cleanup of the evicted provider's httpx pool. The
        # SDK won't do it for us (we pass an explicit ``http_client``).
        # Best-effort — bare ``except RuntimeError`` covers the rare
        # case where the registry is exercised outside a running event
        # loop (e.g. early startup); GC will reclaim the connections
        # eventually but with the ``ResourceWarning`` we'd see in
        # asyncio debug mode.
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(evicted.aclose())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except RuntimeError:
            pass
    return provider


def _resolve_openai_api_key(tenant_config: object | None) -> str:
    """Tenant override first, then ``OPENAI_API_KEY`` env. Empty string if neither.

    Tenant config is duck-typed (loose ``getattr`` protocol) so callers
    can pass any object exposing an ``openai_api_key`` attribute (or
    none at all).
    """
    if tenant_config is not None:
        key = getattr(tenant_config, "openai_api_key", None)
        if key:
            return key
    return os.environ.get("OPENAI_API_KEY", "")


def get_embedding_provider(
    name: str,
    tenant_config: object | None = None,
) -> EmbeddingProvider:
    """Construct an embedding provider by name.

    Parameters
    ----------
    name:
        Provider identifier: ``"openai"``, ``"local"``, or ``"fake"``.
    tenant_config:
        Optional ``ResolvedConfig``-shaped object for per-tenant overrides
        (``openai_api_key``, ``embedding_model``). Can be ``None`` for
        platform-only callers (e.g. core-worker).

    Raises
    ------
    ValueError
        If the provider name is unknown.
    """
    if name == ProviderName.FAKE:
        return FakeEmbeddingProvider()

    if name == ProviderName.OPENAI:
        api_key = _resolve_openai_api_key(tenant_config)
        if not api_key:
            # Lazy import — avoids a circular at module-load time and
            # keeps platform setup off the cold-start critical path
            # for callers that don't need it.
            from common.embedding._platform import get_platform_embedding

            platform = get_platform_embedding()
            if platform is not None:
                logger.info(
                    "No tenant key for OpenAI embedding, using platform embedding (%s)",
                    platform.model,
                )
                return platform
            logger.warning(
                "No API key for OpenAI embedding provider, returning FakeEmbeddingProvider",
            )
            return FakeEmbeddingProvider()
        embed_model = (
            getattr(tenant_config, "embedding_model", None)
            if tenant_config is not None
            else None
        ) or OPENAI_EMBEDDING_MODEL
        return _get_or_create_openai_provider(api_key, embed_model)

    if name == ProviderName.LOCAL:
        return LocalEmbedding()

    raise ValueError(f"Unknown embedding provider: {name}")
