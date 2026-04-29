import asyncio
import logging
import os as _os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp as ASGIApplication
from starlette.types import Receive, Scope, Send

from common.structlog_config import configure_logging
from core_api.config import settings as app_settings

# Must run before any other module-level `logging.getLogger(...)` call emits
# a record, otherwise those records end up going through stdlib's default
# handler instead of our JSON/GCP pipeline.
configure_logging(
    app_settings.environment,
    app_settings.log_level,
    json_logs=app_settings.log_format_json,
    log_file=app_settings.log_file or None,
)

logger = logging.getLogger(__name__)

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from common.events.factory import get_event_bus
from core_api.clients.storage_client import get_storage_client
from core_api.constants import VERSION, is_mcp_path
from core_api.consumer import register_consumers
from core_api.mcp_server import get_mcp_app, mcp_lifespan
from core_api.middleware.per_tenant_concurrency import per_tenant_storage_slot
from core_api.middleware.rate_limit import limiter
from core_api.middleware.request_timeout import (
    _TIMEOUT_OPT_OUT_PATHS,
    RequestTimeoutMiddleware,
)
from core_api.routes.agents import router as agents_router
from core_api.routes.audit import router as audit_router
from core_api.routes.crystallizer import router as crystallizer_router
from core_api.routes.documents import router as documents_router
from core_api.routes.entities import router as entities_router
from core_api.routes.evolve import router as evolve_router
from core_api.routes.fleet import router as fleet_router
from core_api.routes.health import router as health_router
from core_api.routes.insights import router as insights_router
from core_api.routes.memories import admin_memories_router
from core_api.routes.memories import router as memories_router
from core_api.routes.plugin import plugin_bootstrap_router
from core_api.routes.plugin import router as plugin_router
from core_api.routes.settings import router as settings_router
from core_api.routes.stats import router as stats_router
from core_api.routes.stm import router as stm_router

_SECURITY_HEADERS = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
    "content-security-policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https://fastapi.tiangolo.com https://avatars.githubusercontent.com; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    ),
}
# Pre-encode once — ASGI headers are list[tuple[bytes, bytes]]
_SECURITY_HEADERS_ENCODED = [(k.encode(), v.encode()) for k, v in _SECURITY_HEADERS.items()]
_SECURITY_HEADER_KEYS = {k.encode() for k in _SECURITY_HEADERS}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware — compatible with mounted raw ASGI apps like MCP."""

    def __init__(self, app: ASGIApplication) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or is_mcp_path(scope["path"]):
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                existing = [(k, v) for k, v in message.get("headers", []) if k not in _SECURITY_HEADER_KEYS]
                message = {**message, "headers": [*existing, *_SECURITY_HEADERS_ENCODED]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


@asynccontextmanager
async def lifespan(app):
    # Increase default thread pool for concurrent LLM calls via asyncio.to_thread()
    import asyncio as _aio
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=100)
    _aio.get_event_loop().set_default_executor(executor)

    # Initialize Sentry error tracking if configured
    if app_settings.sentry_dsn:
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=app_settings.sentry_dsn,
                environment=app_settings.environment,
                traces_sample_rate=0.1,
                profiles_sample_rate=0.1,
            )
            logger.info("Sentry initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed, skipping Sentry init")

    # CAURA-595: bridge ``settings.<KEY>`` credential values into
    # ``os.environ`` so the shared ``common.llm._credentials`` and
    # ``common.llm._platform`` modules — which read ``os.environ``
    # directly so core-worker doesn't depend on pydantic-settings —
    # see ``.env``-loaded values too. Must run BEFORE
    # ``init_platform_providers()`` (which reads
    # ``PLATFORM_LLM_API_KEY`` from ``os.environ``).
    from core_api.config import bridge_credentials_to_environ

    bridge_credentials_to_environ()

    # Initialize platform default providers (Caura API keys for tenants without credentials)
    # Placed after Sentry so init exceptions are captured.
    from core_api.providers._platform import init_platform_providers

    init_platform_providers()

    # Fail-fast: validate production environment
    if app_settings.environment == "production":
        if app_settings.is_standalone:
            raise RuntimeError(
                "IS_STANDALONE=true is not allowed in production. "
                "Set IS_STANDALONE=false for production deployments."
            )
        if not app_settings.settings_encryption_key:
            raise RuntimeError(
                "SETTINGS_ENCRYPTION_KEY must be set when ENVIRONMENT=production. "
                'Generate one with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
            )
        _dangerous = {
            "postgres_password": "changeme",
            "jwt_secret": "change-me-in-production",
        }
        for var, bad_val in _dangerous.items():
            val = getattr(app_settings, var, None)
            # SecretStr fields (e.g. postgres_password) wrap the value;
            # unwrap before comparing so the guard isn't silently bypassed.
            if hasattr(val, "get_secret_value"):
                val = val.get_secret_value()
            if val == bad_val:
                raise RuntimeError(f"{var.upper()} must be changed from default for production")
        if not app_settings.admin_api_key:
            raise RuntimeError("ADMIN_API_KEY must be set for production")

    async with mcp_lifespan():
        # Standalone mode: initialise fixed tenant id
        if app_settings.is_standalone:
            from core_api.standalone import init_standalone

            init_standalone()

        # Backfill agent rows for any memories written before agent tracking
        try:
            from core_api.db.session import async_session
            from core_api.services.agent_service import backfill_agents

            async with async_session() as db:
                count = await backfill_agents(db)
                if count:
                    await db.commit()
                    print(f"[startup] Backfilled {count} agent(s) from memories")
        except Exception as e:
            print(f"[startup] Agent backfill skipped: {e}")

        # Wire service hooks (audit + recall tracking)
        from core_api.repositories import memory_repo
        from core_api.services.audit_service import log_action
        from core_api.services.hooks import ServiceHooks, configure_hooks

        configure_hooks(
            ServiceHooks(
                audit_log=log_action,
                on_recall=memory_repo.increment_recall,
            )
        )

        # CAURA-628: bind + start the audit batch flusher. ``log_action``
        # checks for an active queue and falls back to a synchronous
        # POST when ``audit_queue_max_size = 0`` (kill-switch) or the
        # queue isn't bound (early startup, tests).
        audit_queue = None
        if app_settings.audit_queue_max_size > 0:
            from core_api.services.audit_queue import (
                AuditEventQueue,
                set_audit_queue,
            )

            async def _flush_one_tenant(tid: str, tevs: list[dict]) -> None:
                # Sentinel bucket for events without a tenant_id — log + skip
                # rather than failing the whole batch (CAURA-631 review).
                if tid == "_UNKNOWN_TENANT_":
                    logger.warning(
                        "audit batch contained %d events with no tenant_id; "
                        "skipping write (events unattributable)",
                        len(tevs),
                    )
                    return
                try:
                    async with per_tenant_storage_slot("storage_write", tid):
                        await get_storage_client().create_audit_logs_bulk(tevs)
                except Exception:
                    logger.exception(
                        "audit batch flush failed for tenant=%s (events=%d); "
                        "events lost from this tenant's slice",
                        tid,
                        len(tevs),
                    )
                    raise

            async def _flush_audit_batch(events: list[dict]) -> None:
                # Group by tenant + flush concurrently with per-tenant storage
                # slot (cap=2) gating each group. Without the slot the flusher
                # hoards storage-writer pool slots while ``/memories/bulk``
                # requests queue, producing the 72% bulk_write 429 spike from
                # loadtest 1777462612 (CAURA-631). Concurrent fan-out keeps
                # flush latency bounded to the slowest tenant rather than
                # serialising over them.
                #
                # Per-tenant failures don't abort sibling tenants;
                # ``return_exceptions=True`` lets every group's outcome land,
                # then we re-raise the last error so ``_drain_and_flush`` can
                # bump ``_failed_count``. Slightly over-reports failures when
                # one tenant succeeds and another fails in the same chunk —
                # acceptable vs silently swallowing errors. Per-tenant detail
                # is logged inside ``_flush_one_tenant`` either way.
                by_tenant: dict[str, list[dict]] = {}
                for ev in events:
                    # ``.get()`` so a malformed event missing tenant_id can't
                    # KeyError out the whole batch — bucket it under the
                    # sentinel and skip the storage write for that bucket.
                    tenant_id = ev.get("tenant_id") or "_UNKNOWN_TENANT_"
                    by_tenant.setdefault(tenant_id, []).append(ev)

                results = await asyncio.gather(
                    *(_flush_one_tenant(tid, tevs) for tid, tevs in by_tenant.items()),
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, BaseException)]
                if errors:
                    # ``with_traceback`` preserves the original failure stack
                    # so APM / Sentry shows the storage-call frame, not just
                    # the re-raise site.
                    last_error = errors[-1]
                    raise last_error.with_traceback(last_error.__traceback__)

            audit_queue = AuditEventQueue(
                max_queue_size=app_settings.audit_queue_max_size,
                flush_threshold=app_settings.audit_queue_flush_threshold,
                flush_interval_seconds=app_settings.audit_queue_flush_interval_seconds,
                flush_callable=_flush_audit_batch,
            )
            set_audit_queue(audit_queue)
            await audit_queue.start()

        # Start lifecycle automation scheduler
        from core_api.services.lifecycle_service import lifecycle_scheduler
        from core_api.tasks import cancel_all_tasks, track_task

        track_task(lifecycle_scheduler())

        # ``register_consumers`` must run before ``bus.start`` — the
        # Pub/Sub backend spawns pull loops from the handler registry
        # snapshot taken at start time, so a late ``subscribe`` would
        # silently orphan the handler. Inprocess mode (tests, OSS
        # standalone) makes ``start`` a no-op so this wiring is
        # harmless there.
        register_consumers()
        event_bus = get_event_bus()
        await event_bus.start()

        yield

        # Each shutdown step is independent — a failure in one (a
        # bus pull-loop close that raises, a tracked task whose
        # cancellation hits a CancelledError swallow somewhere,
        # an httpx pool already closed) must not skip the rest, or
        # we leak the resources the later steps would have freed.
        # Wrap each in its own try/except and continue; the
        # executor.shutdown at the end always runs.
        #
        # Order matters: drain the audit queue BEFORE closing the
        # storage client — the final flush goes through that client.
        # Bus stop also happens before storage-client close because
        # the bus's pull-loops may still be issuing storage calls
        # mid-cancel.
        shutdown_steps: list = []
        if audit_queue is not None:
            shutdown_steps.append(audit_queue.stop(timeout=5.0))
        shutdown_steps.extend(
            [
                event_bus.stop(),
                cancel_all_tasks(),
                get_storage_client().close(),
            ]
        )
        for coro in shutdown_steps:
            try:
                await coro
            except Exception:
                logger.exception("error during shutdown step")
        executor.shutdown(wait=False)


app = FastAPI(
    title="MemClaw",
    version=VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# slowapi reads limiter + handler from app.state; decorators in
# middleware/rate_limit.py consult this at request time.
# SlowAPIMiddleware emits X-RateLimit-Limit/Remaining + Retry-After on
# every response so clients can back off before hitting 429.
app.state.limiter = limiter


async def _json_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Custom handler so 429 bodies match the rest of the API's
    # `{"detail": ...}` envelope. Re-runs slowapi's header injector so
    # X-RateLimit-* + Retry-After still land on the response.
    response = JSONResponse(
        {"detail": f"Rate limit exceeded: {exc.detail}. Try again later."},
        status_code=429,
    )
    response = request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)
    return response


app.add_exception_handler(RateLimitExceeded, _json_rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

if app_settings.is_standalone:
    from core_api.middleware.standalone_tenant import StandaloneTenantMiddleware

    app.add_middleware(StandaloneTenantMiddleware)

app.add_middleware(
    RequestTimeoutMiddleware,
    timeout_seconds=app_settings.request_timeout_seconds,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in app_settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    detail = str(exc) if app_settings.environment != "production" else "Internal Server Error"
    content: dict = {
        "detail": detail,
        "path": request.url.path,
    }
    if app_settings.environment != "production":
        content["error_type"] = type(exc).__name__
    return JSONResponse(status_code=500, content=content)


app.include_router(health_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(admin_memories_router, prefix="/api/v1")
app.include_router(entities_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")
app.include_router(agents_router, prefix="/api/v1")
app.include_router(fleet_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(crystallizer_router, prefix="/api/v1")
app.include_router(plugin_router, prefix="/api/v1")
# Bootstrap aliases — see plugin.py:plugin_bootstrap_router for rationale.
app.include_router(plugin_bootstrap_router, prefix="/api")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(stm_router, prefix="/api/v1")
app.include_router(insights_router, prefix="/api/v1")
app.include_router(evolve_router, prefix="/api/v1")

# Test-only endpoints (time-warp, etc.) — only registered when TESTING=1
if _os.getenv("TESTING") == "1":
    from core_api.routes.testing import router as testing_router

    app.include_router(testing_router, prefix="/api/v1")

app.mount("/mcp", get_mcp_app())


# CAURA-602: turn a silent regression into a startup crash. The
# request-timeout middleware skips a hardcoded path-allowlist; if a
# router prefix or path ever moves and the allowlist isn't updated to
# match, the silent-create class would re-emerge with no error. Verify
# at import time that every opt-out path is actually mounted on this
# app — string-matching is forced by the ASGI scope shape, but at
# least the divergence will fail loudly. ``getattr`` filters out
# ``Host`` route entries (no ``.path``); ``Mount`` and ``APIRoute``
# both carry it.
_registered_paths = {getattr(r, "path", None) for r in app.routes if getattr(r, "path", None)}
for _opt_out in _TIMEOUT_OPT_OUT_PATHS:
    if _opt_out not in _registered_paths:
        raise RuntimeError(
            f"RequestTimeoutMiddleware opt-out path {_opt_out!r} is not "
            "registered on the FastAPI app. Either the route was renamed/"
            "removed or _TIMEOUT_OPT_OUT_PATHS in middleware/request_timeout.py "
            "is stale; both are silent-create regressions waiting to happen."
        )


_static = Path(__file__).resolve().parent.parent.parent / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

# Frontend is served by separate containers (site + app-frontend).
# Nginx gateway handles path-based routing to the correct service.
