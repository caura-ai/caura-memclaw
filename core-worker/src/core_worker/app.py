"""FastAPI application for core-worker.

Consumer-only service — no business HTTP routes, just ``/healthz`` +
``/readyz`` so Cloud Run / k8s can probe liveness and readiness.

Lifespan ordering:
1. ``configure_logging`` reads env BEFORE any module that emits log records.
2. ``init_platform_embedding`` materialises the platform-tier provider
   singleton from ``PLATFORM_EMBEDDING_*`` env vars. Done eagerly so a
   bad config fails the readiness probe instead of nacking every event.
3. ``register_consumers`` wires ``handle_embed_request`` against
   ``Topics.Memory.EMBED_REQUESTED``.
4. ``bus.start()`` spawns Pub/Sub pull loops (no-op for inprocess bus).

Shutdown reverses: stop the bus (drains in-flight messages), close the
shared httpx client, log.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException

from common.embedding import init_platform_embedding
from common.events.base import EventBus
from common.events.factory import get_event_bus
from common.structlog_config import configure_logging
from core_worker.clients.storage_client import close_storage_client, get_storage_client
from core_worker.config import Settings
from core_worker.consumer import configure as configure_consumer
from core_worker.consumer import register_consumers

settings = Settings()  # type: ignore[call-arg]

# Configure logging before any imports emit log records.
configure_logging(
    settings.environment,
    settings.log_level,
    json_logs=settings.log_format_json,
    log_file=settings.log_file or None,
)

logger = logging.getLogger(__name__)

_event_bus: EventBus | None = None


async def _is_ready() -> bool:
    """Readiness check: bus singleton present + still healthy.

    `bus.is_healthy` flips False on the Pub/Sub backend when any pull
    loop has halted on a permanent error (subscription missing, SA
    permission revoked) — without this check, a misconfigured pod
    stays "ready" while silently dropping every inbound event.
    """
    return _event_bus is not None and _event_bus.is_healthy


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global _event_bus

    logger.info("Starting core-worker", extra={"environment": settings.environment})

    # Eager platform-embedding init so a bad PLATFORM_EMBEDDING_* config
    # fails the readiness probe (handle_embed_request returns early when
    # the singleton is None — CAURA-594 worker is platform-only by design).
    init_platform_embedding()

    # Bind the consumer to its dependencies before subscribing — the
    # subscribe must happen BEFORE bus.start() because the Pub/Sub
    # backend spawns pull loops based on the current handler registry.
    configure_consumer(get_storage_client)
    register_consumers()

    _event_bus = get_event_bus()
    await _event_bus.start()

    yield

    logger.info("Shutting down core-worker")
    if _event_bus is not None:
        await _event_bus.stop()
        _event_bus = None
    await close_storage_client()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MemClaw Core Worker",
        description=(
            "Async-embed event consumer.\n\n"
            "Subscribes to ``Topics.Memory.EMBED_REQUESTED`` and PATCHes "
            "computed embeddings back to core-storage-api. Exposes only "
            "health endpoints."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        # Kubernetes and Cloud Run readiness probes gate on the HTTP
        # status code — a 2xx body of ``{"status": "not_ready"}`` would
        # leave the pod in rotation. Raise 503 so the load balancer
        # actually drains a misconfigured / bus-halted pod.
        if not await _is_ready():
            raise HTTPException(status_code=503, detail="not_ready")
        return {"status": "ok"}

    return app


app = create_app()
