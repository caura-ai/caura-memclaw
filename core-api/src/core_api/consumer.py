"""Pub/Sub consumer for core-api.

Subscribes core-api to ``Topics.Memory.ENRICHED`` (CAURA-595): the
back-channel published by core-worker after a successful enrichment
PATCH triggers contradiction detection on the async write path.

Atomic-fact fan-out (parent → child memories) is intentionally not
handled here. The worker drops ``EnrichmentResult.atomic_facts``
entirely (see ``_ENRICHMENT_UNROUTED_FIELDS`` in
``core-worker/src/core_worker/consumer.py``), so they never reach
storage. Persisting them at the worker side and fanning out from a
storage fetch is a separate piece of work.

Race with the embed worker
--------------------------
``MemoryEnriched`` does not carry the parent embedding — the consumer
fetches the memory row from storage to recover it. The embed worker
runs independently of the enrich worker; if enrichment finishes first,
the row's ``embedding`` column is still NULL on read-back. The handler
logs the deferred state with a ``deferred_reason`` field so Cloud
Logging metrics can quantify it, then ack-completes. A symmetric
``Topics.Memory.EMBEDDED`` consumer that fires the same detection on
the other ordering would close the gap; until that exists,
contradiction detection is silently skipped on enrich-first writes.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from common.events.base import Event
from common.events.factory import get_event_bus
from common.events.memory_enriched import MemoryEnriched
from common.events.topics import Topics
from core_api.clients.storage_client import get_storage_client
from core_api.services.contradiction_detector import detect_contradictions_async

logger = logging.getLogger(__name__)


async def handle_memory_enriched(event: Event) -> None:
    """Process one ``Topics.Memory.ENRICHED`` event.

    Loads the memory row from storage (so we have the embedding +
    fleet_id that the minimal payload doesn't carry) and dispatches to
    ``detect_contradictions_async`` when the embedding has already
    landed. Schema-drift / malformed payloads ack-drop with a loud
    ``dropped=True`` log entry so a poison message can't loop the
    subscription.

    The detection coroutine is awaited inline rather than spawned via
    ``track_task``: it owns its own ``try/except`` and never raises, so
    the bus always acks regardless of detection outcome — but the await
    delays the ack until detection completes, which prevents
    redelivery from stacking concurrent detections on the same memory.
    """
    try:
        payload = MemoryEnriched(**event.payload)
    except (ValidationError, TypeError):
        # ``ValidationError`` covers shape drift (missing required
        # fields, wrong types). ``TypeError`` covers ``event.payload``
        # not being a mapping at all — ``**non_dict`` raises before
        # Pydantic ever sees it. Both are poison-message conditions
        # that re-raising would just nack-loop on; ack-drop loudly.
        logger.exception(
            "dropping malformed memory-enriched payload",
            extra={
                "event_type": event.event_type,
                "event_id": str(event.event_id),
                "dropped": True,
            },
        )
        return

    sc = get_storage_client()
    memory = await sc.get_memory(str(payload.memory_id))
    if memory is None:
        # Row was deleted between the worker's PATCH and our handler
        # picking up the event. Common-enough race to ack-drop without
        # noise; matches the worker's 404 handling.
        logger.info(
            "memory-enriched: target row missing; ack-dropping",
            extra={
                "memory_id": str(payload.memory_id),
                "tenant_id": payload.tenant_id,
            },
        )
        return

    embedding = memory.get("embedding")
    if not embedding:
        # Embed worker hasn't completed yet (or its PATCH 404'd).
        # ``deferred_reason`` is structured so a Cloud Logging metric
        # can scrape the count of skipped detections — without it,
        # there's no production visibility into how often the race
        # fires.
        logger.info(
            "memory-enriched: embedding not yet present; deferring contradiction detection",
            extra={
                "memory_id": str(payload.memory_id),
                "tenant_id": payload.tenant_id,
                "deferred_reason": "embedding_missing",
            },
        )
        return

    fleet_id = memory.get("fleet_id")

    # Pass the already-fetched row through so the detector skips a
    # redundant ``sc.get_memory`` (it still re-checks ``deleted_at``
    # for the soft-delete-during-detection race).
    await detect_contradictions_async(
        payload.memory_id,
        payload.tenant_id,
        fleet_id,
        payload.content,
        embedding,
        new_memory=memory,
    )

    logger.info(
        "memory-enriched processed",
        extra={
            "memory_id": str(payload.memory_id),
            "tenant_id": payload.tenant_id,
            "fleet_id": fleet_id,
            "embedding_dim": len(embedding),
        },
    )


def register_consumers() -> None:
    """Wire the consumers into the event bus.

    Must run before ``bus.start()`` — the Pub/Sub backend spawns its
    pull loops in ``start()`` from the current handler registry, so a
    late ``subscribe()`` would silently orphan the handler.
    """
    bus = get_event_bus()
    bus.subscribe(Topics.Memory.ENRICHED, handle_memory_enriched)
