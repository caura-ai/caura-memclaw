"""`get_event_bus()` — single resolver for the active backend.

Reads `EVENT_BUS_BACKEND` (default `inprocess`) plus Pub/Sub-specific env
vars for `pubsub` mode. Returns a singleton so publishers and
subscribers in the same process share the same bus instance.
"""

from __future__ import annotations

import os
import threading

from common.events.base import EventBus
from common.events.inprocess import InProcessEventBus
from common.events.pubsub import PubSubEventBus

_bus: EventBus | None = None
_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the process-wide event bus, constructing on first call.

    Raises ValueError for unknown backend names and RuntimeError when
    Pub/Sub is requested without the required env vars set — upstream
    services should validate these at boot rather than see an obscure
    failure on first publish.
    """
    global _bus
    if _bus is not None:
        return _bus
    with _lock:
        if _bus is not None:
            return _bus
        backend = os.getenv("EVENT_BUS_BACKEND", "inprocess").lower()
        if backend == "inprocess":
            _bus = InProcessEventBus()
        elif backend == "pubsub":
            project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("EVENT_BUS_PROJECT_ID")
            sub_prefix = os.getenv("EVENT_BUS_SUBSCRIPTION_PREFIX")
            if not project_id or not sub_prefix:
                raise RuntimeError(
                    "EVENT_BUS_BACKEND=pubsub requires GCP_PROJECT_ID (or "
                    "EVENT_BUS_PROJECT_ID) and EVENT_BUS_SUBSCRIPTION_PREFIX"
                )
            # Fail fast at factory resolution (service boot) rather than on
            # first publish, so a misconfigured deploy that's missing the
            # Pub/Sub SDK surfaces immediately instead of after the first
            # real event.
            PubSubEventBus._ensure_pubsub_sdk()
            _bus = PubSubEventBus(project_id, sub_prefix)
        else:
            raise ValueError(
                f"Unknown EVENT_BUS_BACKEND {backend!r}; expected inprocess or pubsub"
            )
        return _bus


async def reset_event_bus_for_testing() -> None:
    """Clear the cached bus so tests can exercise different backends or
    isolate state between cases. Production code must NOT call this.

    Async so we can ``await current.stop()`` before discarding — without
    this, a ``PubSubEventBus`` would leak its ``ThreadPoolExecutor`` +
    gRPC ``SubscriberClient`` each test, and ``InProcessEventBus`` would
    leave pending asyncio tasks behind. Callers must use ``await`` (test
    fixtures that used the sync form need updating).
    """
    global _bus
    with _lock:
        current = _bus
        _bus = None
    if current is not None:
        await current.stop()
