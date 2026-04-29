"""Embedding-provider factory.

Resolves a concrete provider instance from the canonical name +
optional tenant config. Tier order:

    Tenant key  →  Platform singleton  →  FakeEmbeddingProvider

Env-driven (no service-config dependency) so both core-api (tenant-aware)
and core-worker (platform-only) can use it.
"""

from __future__ import annotations

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
# process; raise if that proves wrong. ``AsyncOpenAI`` doesn't require
# explicit ``aclose`` — the GC drops the underlying httpx pool when
# the evicted instance has no remaining refs.
_OPENAI_CACHE_MAX = 256
_openai_provider_cache: OrderedDict[
    tuple[str, str, str | None, bool, str | None, int | None],
    OpenAIEmbeddingProvider,
] = OrderedDict()


def _get_or_create_openai_provider(
    api_key: str,
    model: str,
    base_url: str | None,
    send_dimensions: bool,
    query_instruction: str | None,
    truncate_to_dim: int | None,
) -> OpenAIEmbeddingProvider:
    """LRU-bounded ``OpenAIEmbeddingProvider`` lookup keyed on the full client
    config tuple.

    Cache key includes ``base_url`` / ``send_dimensions`` / ``query_instruction``
    / ``truncate_to_dim`` so the same api_key can host multiple simultaneous
    providers — e.g. real OpenAI for one tenant and a local TEI sidecar with
    a Qwen3 instruction prefix for another — without the second silently
    aliasing to the first cached client.

    Not strictly async-safe across concurrent cache misses for the same
    key — two coroutines can both observe a miss before either inserts.
    The race is harmless: the later insert overwrites the earlier with
    a functionally identical client (same key tuple, no per-call state),
    and the earlier instance gets dropped on the next GC. Not worth an
    asyncio.Lock for the cost of a duplicated TLS handshake on the
    first concurrent miss.
    """
    cache_key = (
        api_key,
        model,
        base_url,
        send_dimensions,
        query_instruction,
        truncate_to_dim,
    )
    cached = _openai_provider_cache.get(cache_key)
    if cached is not None:
        _openai_provider_cache.move_to_end(cache_key)
        return cached
    provider = OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        send_dimensions=send_dimensions,
        query_instruction=query_instruction,
        truncate_to_dim=truncate_to_dim,
    )
    _openai_provider_cache[cache_key] = provider
    if len(_openai_provider_cache) > _OPENAI_CACHE_MAX:
        _openai_provider_cache.popitem(last=False)
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
        base_url = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or None
        send_dimensions = (
            os.environ.get("OPENAI_EMBEDDING_SEND_DIMENSIONS", "true").lower()
            != "false"
        )
        # Option B: instruction-aware query encoding. Default applied to all
        # /api/v1/search calls; documents (writes) embed unmodified text.
        query_instruction = (
            os.environ.get("EMBEDDING_QUERY_INSTRUCTION") or None
        )
        # Matryoshka truncation: lets us run instruction-aware models with
        # native dim > VECTOR_DIM (Qwen3-Embedding-0.6B is 1024-d, 4B is
        # 2560-d, 8B is 4096-d) against an unchanged 768-d schema. The
        # ``OpenAIEmbeddingProvider._postprocess`` slices and L2-renormalizes
        # so cosine sim correctness is preserved.
        truncate_raw = os.environ.get("OPENAI_EMBEDDING_TRUNCATE_TO_DIM")
        truncate_to_dim = int(truncate_raw) if truncate_raw else None
        return _get_or_create_openai_provider(
            api_key,
            embed_model,
            base_url,
            send_dimensions,
            query_instruction,
            truncate_to_dim,
        )

    if name == ProviderName.LOCAL:
        return LocalEmbedding()

    raise ValueError(f"Unknown embedding provider: {name}")
