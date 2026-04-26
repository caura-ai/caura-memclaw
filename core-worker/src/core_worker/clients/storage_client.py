"""Async HTTP client for core-storage-api.

The worker hits a small set of endpoints:
* ``PATCH /api/v1/storage/memories/{memory_id}`` — write the embedding
  (CAURA-594) or enrichment fields (CAURA-595) back once the provider
  call succeeds.
* ``GET /api/v1/storage/memories/embedding-by-content-hash`` — best-
  effort cache lookup so an identical-content row that already has an
  embedding short-circuits the embed provider call.

Module-level singleton client (``get_storage_client()``) so connection
pooling persists across pull-loop iterations. Closed via
:func:`close_storage_client` from the FastAPI lifespan.

CAURA-591 Y3 / CAURA-595 follow-up: every outbound request is signed
with a Cloud Run ID token whose audience is the storage service URL.
The identity_token helper transparently returns ``{}`` when no creds
are available (local dev / OSS) so the unauthenticated docker-compose
path continues to work; on Cloud Run the metadata server issues a
real token and the writer's ``--no-allow-unauthenticated`` accepts it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import httpx

from core_worker.clients.identity_token import evict as _evict_id_token
from core_worker.clients.identity_token import fetch_auth_header

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
# Audience for the writer ID token — set alongside the singleton in
# ``get_storage_client()``. Mirrors core-api's ``self._base_url``: the
# rstrip'd ``core_storage_api_url`` from settings, used both as the
# httpx ``base_url`` and as the Cloud Run audience claim.
_audience: str | None = None
_PREFIX = "/api/v1/storage"


def get_storage_client() -> httpx.AsyncClient:
    """Return the singleton :class:`httpx.AsyncClient` used by the consumer.

    Settings are read from a freshly-built ``Settings()`` on first call
    only — subsequent calls return the cached instance regardless of
    any env changes. Tests that need a different config call
    :func:`close_storage_client` between cases to force re-construction.
    """
    global _client, _audience
    if _client is None:
        # Lazy import — avoids paying the pydantic-settings load cost
        # on cold-start for the worker's healthz probe.
        from core_worker.config import Settings

        s = Settings()  # type: ignore[call-arg]
        # ``rstrip("/")`` so a trailing slash in the env var doesn't
        # produce a different audience string than Cloud Run validates.
        _audience = s.core_storage_api_url.rstrip("/")
        _client = httpx.AsyncClient(
            base_url=_audience,
            timeout=httpx.Timeout(s.storage_http_timeout_s),
        )
    return _client


async def close_storage_client() -> None:
    """Close the singleton (idempotent). Called from app lifespan shutdown."""
    global _client, _audience
    if _client is not None:
        await _client.aclose()
        _client = None
        _audience = None


async def _signed_call(
    fn: Callable[..., Awaitable[httpx.Response]],
    *args: Any,
    **kwargs: Any,
) -> httpx.Response:
    """Issue ``fn(*args, **kwargs)`` with the Cloud Run ID-token
    Authorization header attached, with one transparent retry on 401.

    ``fn`` is a bound method on the caller's ``httpx.AsyncClient``
    (``client.get``, ``client.patch``, etc.) — single integration point
    so a future endpoint helper can't add a request without auth or
    skip the eviction-on-401 path.

    On 401: evict the cached token (so a mid-TTL credential rotation or
    SA/binding fix self-heals instead of waiting out the 50 min cache),
    fetch a fresh token, and retry the request once. Without this
    retry, every in-flight task during a rotation burns one Pub/Sub
    delivery attempt against the DLQ budget; the retry makes a rotation
    invisible to callers. A second 401 (rotation didn't help, or the
    SA was actually unbound) propagates to the caller's normal
    error-handling path.

    On the OSS / local-dev path (``_audience`` unset, or google.auth
    unavailable) the merged auth header is empty and the call goes
    out unauthenticated — matching the docker-compose ``allUsers``
    writer. 403 is intentionally NOT a trigger for eviction: 403 means
    the token was accepted but IAM rejected the caller, which the
    cache cannot fix.
    """
    headers = dict(kwargs.pop("headers", None) or {})
    if _audience is not None:
        headers.update(await fetch_auth_header(_audience))
    resp = await fn(*args, headers=headers, **kwargs)
    if resp.status_code == 401 and _audience is not None:
        _evict_id_token(_audience)
        # Rebind to a NEW dict rather than ``.update()`` in place — the
        # first call still holds a reference to the original headers
        # mapping (httpx is already done with it, but a debug logger
        # or test inspector that snapshots the call args would
        # otherwise see the rotated token retro-fitted into the first
        # call's record).
        headers = {**headers, **await fetch_auth_header(_audience)}
        resp = await fn(*args, headers=headers, **kwargs)
    return resp


async def find_embedding_by_content_hash(
    client: httpx.AsyncClient,
    *,
    tenant_id: str,
    content_hash: str,
) -> list[float] | None:
    """Look up a previously computed embedding for the same content.

    Returns ``None`` when there's no cached row, the row exists but its
    embedding is NULL (still pending), or the storage call fails for
    any reason. Callers fall through to the embedding provider.

    Storage contract (``GET /memories/embedding-by-content-hash``)
    returns the bare ``list[float] | None`` — JSON ``null`` on miss,
    not ``{"embedding": null}``. Treating the body as a dict and
    calling ``.get("embedding")`` on a JSON-null parse crashes with
    ``AttributeError: 'NoneType' object has no attribute 'get'``,
    which used to be masked by 403s before the CAURA-595 ID-token
    PR landed and the cache lookup actually started succeeding.
    """
    try:
        resp = await _signed_call(
            client.get,
            f"{_PREFIX}/memories/embedding-by-content-hash",
            params={"tenant_id": tenant_id, "content_hash": content_hash},
        )
        if resp.status_code != 200:
            return None
        # ``resp.json()`` raises on malformed bodies (a partial response
        # from a stray proxy, an HTML error page sneaking through with
        # a 200, etc.). The docstring promises "fails for any reason →
        # None" so the worker falls through to a fresh provider call
        # instead of nacking and forcing Pub/Sub redelivery — staying
        # consistent with the ``HTTPError`` handling above.
        return resp.json()
    except (httpx.HTTPError, ValueError):
        logger.exception("storage embedding-cache lookup failed; treating as miss")
        return None


async def update_memory_embedding(
    client: httpx.AsyncClient,
    *,
    memory_id: UUID,
    tenant_id: str,
    embedding: list[float],
) -> None:
    """PATCH the embedding onto the memory row.

    Raises on transient (5xx / 429) responses so the consumer nacks
    → Pub/Sub redelivers. Permanent 4xx errors (404, 422) are logged
    and ack-dropped — retrying them just burns the
    max-delivery-attempts budget against an event that will never
    succeed (row deleted between event publish and processing, or a
    schema mismatch the storage layer rejects either way).

    ``tenant_id`` is included in the body so the storage layer can
    guard against cross-tenant updates.

    The hot-path writer set ``metadata.embedding_pending=true`` when
    deferring the embed; clear it via ``metadata_patch`` on the same
    PATCH so a read-after-success returns clean state instead of
    confusingly claiming the row is still pending.
    """
    resp = await _signed_call(
        client.patch,
        f"{_PREFIX}/memories/{memory_id}",
        json={
            "tenant_id": tenant_id,
            "embedding": embedding,
            "metadata_patch": {"embedding_pending": False},
        },
    )
    if resp.status_code == 404:
        # Row was deleted after the event was published — common-enough
        # legitimate race that it shouldn't churn the DLQ.
        logger.warning(
            "memory %s not found in storage; ack-dropping embed-request "
            "(row deleted between publish and processing)",
            memory_id,
        )
        return
    if resp.status_code == 422:
        # Storage rejected the payload shape. Won't pass on a redelivery
        # either — log loudly so the schema mismatch is visible and ack.
        logger.warning(
            "storage rejected embedding PATCH for memory %s with 422; ack-dropping",
            memory_id,
        )
        return
    # 5xx / 429 / network → raise → consumer nacks → Pub/Sub redelivers.
    resp.raise_for_status()


async def update_memory_enrichment(
    client: httpx.AsyncClient,
    *,
    memory_id: UUID,
    tenant_id: str,
    fields: dict,
) -> None:
    """PATCH enrichment-output fields onto the memory row.

    Mirrors :func:`update_memory_embedding` for the
    ``Topics.Memory.ENRICH_REQUESTED`` worker path. ``fields`` is the
    pre-validated dict of column→value pairs the consumer wants to write
    (memory_type, weight, title, summary, tags, status, ts_valid_*,
    contains_pii, pii_types, retrieval_hint). The storage-side
    ``memory_update`` filters by ``hasattr(Memory, key)`` so unknown keys
    are silently dropped — caller still validates upstream against
    ``EnrichmentResult`` so a typo surfaces as a Pydantic error, not a
    silent no-op.

    Idempotency: at-least-once delivery means a redelivered enrich event
    re-PATCHes with the (deterministic up to LLM noise) same fields.
    Safe — the storage update is a write-after-write.

    Same 404/422 short-circuits as :func:`update_memory_embedding`.
    """
    if not fields:
        # Defensive — heuristic fallback may produce an all-defaults
        # result; an empty PATCH is wasted work and noisy in storage logs.
        logger.debug("skipping empty enrichment PATCH for memory %s", memory_id)
        return
    resp = await _signed_call(
        client.patch,
        f"{_PREFIX}/memories/{memory_id}",
        json={"tenant_id": tenant_id, **fields},
    )
    if resp.status_code == 404:
        logger.warning(
            "memory %s not found in storage; ack-dropping enrich-request "
            "(row deleted between publish and processing)",
            memory_id,
        )
        return
    if resp.status_code == 422:
        logger.warning(
            "storage rejected enrichment PATCH for memory %s with 422; ack-dropping",
            memory_id,
        )
        return
    resp.raise_for_status()
