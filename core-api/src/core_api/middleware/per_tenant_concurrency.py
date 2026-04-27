"""Per-tenant in-flight concurrency cap.

The slowapi-based ``rate_limit`` middleware bounds request *rate*. A
single tenant burst can still occupy every worker slot in a container
and starve other tenants — exactly the noisy-neighbor pattern the
loadtest harness flagged. Enforces a per-tenant cap as an
``asyncio.Semaphore``: when full, requests fail fast with 429 instead
of queueing until they time out at the worker layer (which surfaces as
5xx to the caller).

State is per-process. With cap ``N`` and Cloud Run ``max_instances``
``M``, the fleet-wide cap is ``N * M`` — size in concert with
``containerConcurrency``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import HTTPException

from core_api.config import settings

logger = logging.getLogger(__name__)

Scope = Literal["write", "search"]

# Per-(scope, tenant) state. Read-modify-write of ``_TENANT_SEMAPHORES``
# in ``_get_semaphore`` is safe without a lock because asyncio runs one
# coroutine at a time on a single event loop and neither ``dict.get``
# nor ``dict.__setitem__`` await — no other task can interleave between
# the miss and the install.
_TENANT_SEMAPHORES: dict[tuple[Scope, str], asyncio.Semaphore] = {}


def _cap_for(scope: Scope) -> int:
    """Resolve the configured cap for ``scope`` from ``Settings``.

    Centralised so callers can't drift — the cap is baked into the
    Semaphore at creation time, so a per-call ``cap`` argument would
    silently bind to the first call's value and ignore subsequent
    setting changes.
    """
    if scope == "write":
        return settings.per_tenant_write_concurrency
    return settings.per_tenant_search_concurrency


def _get_semaphore(scope: Scope, tenant_id: str) -> asyncio.Semaphore:
    """Return (and lazily create) the semaphore for ``(scope, tenant_id)``.

    The dict grows monotonically with the set of tenants the instance
    has seen — bounded by the active tenant count, never reset within
    a process lifetime.
    """
    key = (scope, tenant_id)
    sem = _TENANT_SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_cap_for(scope))
        _TENANT_SEMAPHORES[key] = sem
    return sem


@asynccontextmanager
async def per_tenant_slot(scope: Scope, tenant_id: str) -> AsyncIterator[None]:
    """Acquire one of the per-tenant slots for ``scope``, or raise 429
    fast.

    Cap is read from ``Settings`` (``per_tenant_write_concurrency`` or
    ``per_tenant_search_concurrency``) at semaphore-creation time.
    Use as a context manager around the request handler body. Releases
    on exit (success or exception) so a handler raising 5xx still
    frees the slot.
    """
    sem = _get_semaphore(scope, tenant_id)
    try:
        async with asyncio.timeout(settings.per_tenant_acquire_timeout_seconds):
            await sem.acquire()
    except TimeoutError:
        logger.info(
            "per-tenant concurrency cap reached",
            extra={"scope": scope, "tenant_id": tenant_id, "cap": _cap_for(scope)},
        )
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent {scope} requests; retry shortly.",
        )
    try:
        yield
    finally:
        sem.release()


def _reset_for_tests() -> None:
    """Drop all tracked semaphores. Test-only — production never calls
    this."""
    _TENANT_SEMAPHORES.clear()
