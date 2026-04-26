"""Idempotency-Key helper (CAURA-601).

Per-route opt-in: handlers call :func:`idempotency_for` after auth
has enforced the tenant, get back a :class:`IdempotencyGuard`, check
``guard.cached_replay`` for a cached response, and call
``guard.record(...)`` after running the work.

Not a middleware — slowapi + FastAPI already handle the request/response
pipeline, and per-handler wiring keeps the replay contract explicit at
the route level. The boilerplate is five lines per decorated handler.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fastapi import HTTPException, Request

from core_api.clients.storage_client import get_storage_client
from core_api.config import settings

logger = logging.getLogger(__name__)

IDEMPOTENCY_HEADER = "Idempotency-Key"
# Matches Stripe's limit. Guards the `Text` column against oversized
# values from buggy or adversarial clients before it reaches storage.
MAX_IDEMPOTENCY_KEY_LEN = 255

# Body-metadata keys checked when the ``Idempotency-Key`` HTTP
# header is absent. ``Idempotency-Key`` (header) remains the
# canonical form per the IETF draft; the body fallback exists for
# SDK / load-test clients that can't easily set headers per request.
# ``idempotency_key`` first, then the legacy ``client_idempotency_key``
# alias the loadtest harness uses.
_IDEMPOTENCY_BODY_FIELDS = ("idempotency_key", "client_idempotency_key")


def idempotency_key_from_metadata(metadata: dict | None) -> str | None:
    """Extract the idempotency key from a request body's ``metadata``
    dict, if present. Returns ``None`` if metadata is missing/empty
    or no recognised key field is set.

    The header takes precedence — call this only as a fallback when
    the header is absent. Routes that wire idempotency wire it once
    via:

        key = idempotency_header or idempotency_key_from_metadata(body.metadata)
    """
    if not metadata:
        return None
    for field in _IDEMPOTENCY_BODY_FIELDS:
        value = metadata.get(field)
        # Return the un-stripped value: ``idempotency_for`` hashes the
        # RAW request bytes for replay-conflict detection, so any
        # normalisation here would let two byte-identical retries —
        # one with surrounding whitespace, one without — hit the same
        # cache bucket but mismatch on body hash, triggering a false
        # 422. The blank guard still rejects pure-whitespace values.
        if isinstance(value, str) and value.strip():
            return value
    return None


class IdempotencyGuard:
    """One per request-with-header. Replay or record, never both."""

    def __init__(
        self,
        tenant_id: str,
        key: str,
        request_hash: str,
        cached: dict | None,
    ) -> None:
        self.tenant_id = tenant_id
        self.key = key
        self.request_hash = request_hash
        self._cached = cached

    @property
    def cached_replay(self) -> tuple[Any, int] | None:
        """``(response_body, status_code)`` if the same key + body was
        served before and is still within TTL; else ``None``."""
        if self._cached is None:
            return None
        return self._cached["response_body"], self._cached["status_code"]

    async def record(self, response_body: Any, status_code: int = 200) -> None:
        """Cache the response body under this key. Failures are logged,
        not raised — the client already has the live response; losing the
        cache only costs a chance at future dedup."""
        expires_at = datetime.now(UTC) + timedelta(seconds=settings.idempotency_ttl_seconds)
        try:
            await get_storage_client().upsert_idempotency(
                tenant_id=self.tenant_id,
                idempotency_key=self.key,
                request_hash=self.request_hash,
                response_body=response_body,
                status_code=status_code,
                expires_at=expires_at.isoformat(),
            )
        except Exception:
            logger.warning("Idempotency record failed (non-critical)", exc_info=True)


async def idempotency_for(
    request: Request,
    tenant_id: str,
    idempotency_key: str | None,
    *,
    source: Literal["header", "body"] = "header",
) -> IdempotencyGuard | None:
    """Look up the inbox for (tenant, key). Returns None if the key
    was absent. On a cache hit with a *different* body hash, raises
    422 — reusing the same key with different content is explicitly
    a client error per the IETF idempotency-key draft.

    Call this inside a route handler *after* auth enforcement, so the
    tenant_id passed here is already vetted.

    ``source`` selects a transport-specific prefix (``"header"`` →
    ``header:``, ``"body"`` → ``body:``) prepended before the lookup
    so the same string value sent via different transports never
    shares a cache bucket. Validation runs against the RAW key the
    caller supplied; the prefix is added only for storage. Defaults
    to ``"header"`` so existing callers (header-only routes) keep
    working without changes.

    Body hashing note: uses raw request bytes. Clients retrying a write
    MUST send byte-identical bodies (same JSON serializer, same key
    ordering). Non-deterministic serialization will falsely flag a
    retry as a conflict. Matches Stripe and the IETF draft.

    Concurrency limitation (tracked as follow-up): this function acquires
    no lock. Two requests with the same key arriving within a few
    milliseconds of each other will both miss the cache, both execute
    the handler, and both persist database rows — the ``record()``
    upsert dedups the STORED RESPONSE (first writer wins on the
    response body), but the write side effects have already happened
    twice. Sequential retries (the primary target use case) are fully
    protected. Truly concurrent idempotent writes need either a
    PostgreSQL advisory lock keyed on ``hashtext(tenant_id||':'||key)``
    spanning the handler, or a "processing" sentinel row inserted here
    before the handler runs.
    """
    if not idempotency_key:
        return None
    if not idempotency_key.strip():
        raise HTTPException(status_code=400, detail="Idempotency-Key must not be blank")
    if len(idempotency_key) > MAX_IDEMPOTENCY_KEY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Idempotency-Key must be at most {MAX_IDEMPOTENCY_KEY_LEN} characters",
        )

    # Format change vs. pre-PR (bare key → ``header:<key>``) orphans any
    # in-flight cache entries at deploy. The 24h TTL absorbs this: the
    # worst case is a duplicate request arriving within 24h of deploy
    # that re-executes instead of replaying — same outcome as a cache
    # miss, never a wrong answer.
    namespaced_key = f"{source}:{idempotency_key}"

    body_bytes = await request.body()
    request_hash = hashlib.sha256(body_bytes).hexdigest()

    cached: dict | None
    try:
        cached = await get_storage_client().get_idempotency(tenant_id, namespaced_key)
    except Exception:
        # Storage unavailable: degrade to "no cache" so writes still go
        # through. A lost dedup opportunity beats a blocked write.
        logger.warning("Idempotency lookup failed (degrading to no-cache)", exc_info=True)
        cached = None

    if cached and cached["request_hash"] != request_hash:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key reused with a different request body",
        )

    return IdempotencyGuard(tenant_id, namespaced_key, request_hash, cached)
