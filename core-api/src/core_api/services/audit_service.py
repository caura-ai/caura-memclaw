"""Audit-event ingestion entrypoint.

``log_action`` is the only public surface — every memory mutation calls
it via the configured ``ServiceHooks`` (see ``core_api.app.lifespan``).

CAURA-628 (2026-04-29): the legacy shape POSTed one audit event per
mutation directly to ``core-storage-api``. Under bulk storms that
created up to N HTTP-POSTs-per-bulk-create on the storage-api connection
pool + serialised tenant B's audit traffic behind tenant A's storm at
the AlloyDB ``audit_log`` table-level write lock. ``log_action`` now
enqueues into a process-local ``AuditEventQueue``; a background flusher
batches events and writes them via ``POST /audit-logs/bulk``.

The synchronous-POST fallback path runs when the queue is not active
(early startup, tests that didn't wire it, intentional kill-switch via
``set_audit_queue(None)``). That fallback preserves the legacy
behaviour byte-for-byte so a queue-side bug can't silently drop audit
events; an operator can fall back to the legacy path during an incident
without a redeploy.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core_api.clients.storage_client import get_storage_client
from core_api.services.audit_queue import get_audit_queue


async def log_action(
    db: AsyncSession,
    *,
    tenant_id: str,
    agent_id: str | None = None,
    action: str,
    resource_type: str,
    resource_id: UUID | None = None,
    detail: dict | None = None,
) -> None:
    """Record an audit event. Async-batched via queue when available;
    falls through to a synchronous POST otherwise.

    ``db`` is unused (audit persistence is owned by the storage layer)
    but kept in the signature for back-compat with the ``ServiceHooks``
    contract — callers pass it through; switching them all to drop the
    arg is a separate, scoped change.
    """
    payload = {
        "tenant_id": tenant_id,
        "agent_id": agent_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": str(resource_id) if resource_id else None,
        "detail": detail,
    }
    queue = get_audit_queue()
    if queue is not None:
        # Non-blocking enqueue. Overflow is mapped to a structured
        # warning + drop counter inside the queue — the request hot
        # path stays fast even when storage-api is degraded.
        queue.enqueue(payload)
        return

    # Fallback: synchronous POST. Same shape as pre-CAURA-628.
    sc = get_storage_client()
    await sc.create_audit_log(payload)
