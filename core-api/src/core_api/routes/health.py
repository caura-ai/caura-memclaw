import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Response
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from common.events.factory import get_event_bus
from core_api.cache import redis_healthy
from core_api.clients.storage_client import get_storage_client
from core_api.config import settings
from core_api.constants import VERSION
from core_api.providers._platform import get_platform_init_errors
from core_api.tools import REGISTRY  # SoT registry — populated at import time

# Upper bound on a single dependency probe. Cloud Run typically gives
# health checks 10-30s before marking unhealthy; we want to fail
# well before that so a stalled backend can't hang the whole probe.
_PROBE_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


@router.get("/version")
async def version():
    return {"version": VERSION}


@router.get("/tool-descriptions")
async def tool_descriptions(enriched: bool = False):
    """Return tool descriptions, derived from the SoT registry.

    Default: ``{name: description}`` (backward compatible).
    With ``?enriched=true``: ``{name: {description, stm_only}}``.

    The registry is the single source of truth — ``stm_only`` is
    derived from ``spec.plugin_exposed`` (inverted).
    """
    if enriched:
        return {
            spec.name: {
                "description": spec.description,
                "stm_only": not spec.plugin_exposed,
            }
            for spec in REGISTRY.values()
        }
    return {spec.name: spec.description for spec in REGISTRY.values()}


@router.get("/health")
async def health(response: Response):
    """Liveness + readiness probe.

    Returns 503 when any required dependency is unavailable so deploy
    gates and Cloud Run health checks can fail-fast on status code alone.
    Required deps:
      - storage (core-storage-api): always
      - redis:                      only when ``settings.redis_url`` is set
                                    (empty url = in-memory fallback, OSS default)
      - event_bus:                  ``InProcessEventBus`` always reports ok;
                                    ``PubSubEventBus`` reports ``unhealthy``
                                    when a pull loop has halted on a permanent
                                    subscription / IAM error.

    Non-critical issues (platform provider init errors) flip status to
    ``"degraded"`` but keep a 200 — the app can still serve requests.
    """
    result: dict[str, Any] = {"status": "ok"}
    unhealthy: list[str] = []

    try:
        sc = get_storage_client()
        # wait_for bounds the probe itself so a stalled storage service
        # (pool exhausted, slow query) can't hang the whole health check.
        await asyncio.wait_for(
            sc.count_all(tenant_id="__health_check__"),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        result["storage"] = "connected"
    except Exception:
        # Never surface str(exc) — httpx errors embed the target URL
        # (and any basic-auth creds in it) into the response body.
        # Full exception is on the server log for operators.
        logger.exception("Storage health check failed")
        result["storage"] = "unreachable"
        unhealthy.append("storage")

    if settings.redis_url:
        try:
            redis_ok = await asyncio.wait_for(redis_healthy(), timeout=_PROBE_TIMEOUT_SECONDS)
        except Exception:
            # Catches asyncio.TimeoutError plus any internal failure in
            # redis_healthy() that escapes its own except guard.
            redis_ok = False
        if redis_ok:
            result["redis"] = "connected"
        else:
            result["redis"] = "unavailable"
            unhealthy.append("redis")
    else:
        result["redis"] = "not configured"

    # Event bus: ``is_healthy`` is a synchronous property (no I/O), so no
    # timeout wrapper needed. InProcessEventBus always returns True (no
    # external failure modes); PubSubEventBus flips False when a pull
    # loop has halted on a permanent error (subscription missing, SA
    # permission revoked) — without this probe, such a pod would stay
    # "ready" while silently dropping every inbound event.
    #
    # ``get_event_bus()`` itself can raise (RuntimeError when Pub/Sub
    # env vars are missing, ValueError for an unknown backend). Wrap
    # consistently with the storage + redis probes above so those
    # misconfigs surface as a structured 503, not a bare 500.
    try:
        bus = get_event_bus()
        if bus.is_healthy:
            result["event_bus"] = "ok"
        else:
            result["event_bus"] = "unhealthy"
            unhealthy.append("event_bus")
    except Exception:
        logger.exception("Event bus health check failed")
        result["event_bus"] = "error"
        unhealthy.append("event_bus")

    init_errors = get_platform_init_errors()
    if init_errors:
        if not unhealthy:
            # 200/degraded path — safe to surface detail since operators
            # need actionable info when everything else is fine.
            result["platform_provider_errors"] = init_errors
            result["status"] = "degraded"
        else:
            # 503 path — don't leak internal SDK messages (hostnames,
            # service URLs) alongside a deploy-gate-visible response.
            logger.warning("Platform init errors alongside dep failures: %s", init_errors)

    if unhealthy:
        result["status"] = "unhealthy"
        result["unhealthy_dependencies"] = unhealthy
        response.status_code = HTTP_503_SERVICE_UNAVAILABLE

    return result
