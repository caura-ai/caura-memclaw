import logging
from typing import Any, Literal
from urllib.parse import quote

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# Postgres connection settings. Canonical env var names follow the
# official ``postgres`` Docker image conventions (POSTGRES_USER,
# POSTGRES_PASSWORD, POSTGRES_DB) so the same ``.env`` works for both
# the database container and the app. Legacy ``ALLOYDB_*`` aliases are
# accepted for back-compat and will be dropped in a future major.
class Settings(BaseSettings):
    postgres_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("POSTGRES_HOST", "ALLOYDB_HOST"),
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias=AliasChoices("POSTGRES_PORT", "ALLOYDB_PORT"),
    )
    postgres_user: str = Field(
        default="memclaw",
        validation_alias=AliasChoices("POSTGRES_USER", "ALLOYDB_USER"),
    )
    postgres_password: SecretStr = Field(
        default=SecretStr("changeme"),
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "ALLOYDB_PASSWORD"),
    )
    postgres_database: str = Field(
        default="memclaw",
        validation_alias=AliasChoices("POSTGRES_DB", "POSTGRES_DATABASE", "ALLOYDB_DATABASE"),
    )
    postgres_use_iam_auth: bool = Field(
        default=False,
        validation_alias=AliasChoices("POSTGRES_USE_IAM_AUTH", "ALLOYDB_USE_IAM_AUTH"),
    )
    postgres_require_ssl: bool = Field(
        default=True,
        validation_alias=AliasChoices("POSTGRES_REQUIRE_SSL", "ALLOYDB_REQUIRE_SSL"),
    )
    api_key: str | None = None  # legacy, deprecated
    admin_api_key: str | None = None
    memclaw_api_key: str | None = None  # Optional: when set, all non-admin requests must present this key
    embedding_provider: str = "openai"  # fake | openai | local
    # When True (default), the inline write path embeds on the request
    # and contradiction detection fires before the response returns —
    # the OSS-friendly shape, no worker fleet required.
    #
    # When False (CAURA-594 SaaS), the row persists with embedding=NULL
    # and the write path publishes ``Topics.Memory.EMBED_REQUESTED`` to
    # the bus; ``core-worker`` consumes the event and PATCHes the row.
    # Semantic search tolerates NULL embeddings via FTS fallback (Step
    # A); exact-hash dedup still runs inline. Set the env override
    # ``EMBED_ON_HOT_PATH=false`` on the SaaS deploy.
    embed_on_hot_path: bool = True
    # CAURA-595 mirror of ``embed_on_hot_path`` for LLM enrichment.
    #
    # When True (OSS default), the strong-write pipeline + single-write
    # fast path + bulk-write all run ``enrich_memory`` inline, blocking
    # the response until the LLM call returns.
    #
    # When False (CAURA-595 SaaS), the row persists with the agent-
    # provided values + schema defaults for ``memory_type`` / ``weight``
    # / ``status``, and the write path publishes
    # ``Topics.Memory.ENRICH_REQUESTED`` to the bus; ``core-worker``
    # consumes the event, runs the enricher, and PATCHes the row. The
    # API response surface drops the LLM-derived fields (``title``,
    # ``summary``, ``tags``, ``retrieval_hint``) — clients that need
    # them must re-fetch after the back-channel ENRICHED event lands.
    # Set the env override ``ENRICH_ON_HOT_PATH=false`` on the SaaS
    # deploy.
    enrich_on_hot_path: bool = True
    # Outer cap on the inline embed+enrich gather in ParallelEmbedEnrich.
    # Was hardcoded at 20.0 — too tight under load once embedding moved
    # off the hot path (CAURA-594) and enrichment LLM became the sole
    # occupant. 35s leaves headroom for nano-class LLM tail latency
    # (typical p95 ~6-12s, plus 2 retries x 1s linear backoff) without
    # breaching the 45s outer request budget. Must stay below
    # ``request_timeout_seconds`` so this fires first.
    enrichment_inline_timeout_seconds: float = 35.0
    # Inner timeout for the optional hint-based re-embed roundtrip after
    # enrichment lands a retrieval_hint. Pure quality-vs-latency knob;
    # only fires when ``embed_on_hot_path`` is True.
    enrichment_hint_reembed_timeout_seconds: float = 10.0
    # Per-call timeout passed to the AsyncOpenAI client (covers both LLM
    # enrichment and embedding providers). Without an explicit value the
    # SDK rides httpx's default — long enough that a single hung upstream
    # call eats the whole enrichment budget silently. 25s gives the
    # provider room to respond while still leaving budget for one retry
    # under the inline ceiling.
    openai_request_timeout_seconds: float = 25.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    entity_extraction_provider: str = "openai"  # none | fake | openai | anthropic | openrouter | gemini
    entity_extraction_model: str = "gpt-5.4-nano"
    use_llm_for_memory_creation: bool = True
    sentry_dsn: str = ""  # Set to enable Sentry error tracking
    redis_url: str = ""  # e.g. redis://localhost:6379/0. Empty = in-memory fallback.
    db_pool_size: int = 50
    db_max_overflow: int = 50
    cors_origins: str = "http://localhost:3000"
    # Request-wide budget enforced by RequestTimeoutMiddleware. 45s fits
    # comfortably under the 60s LB cap and 300s Cloud Run platform ceiling,
    # so a hung handler cannot keep a request slot past this.
    #
    # Residual risk: asyncio.timeout cancels the coroutine task but cannot
    # cancel sync threads started via asyncio.to_thread (Vertex / Gemini
    # provider SDKs). A hung provider holds its ThreadPoolExecutor slot
    # past the 504; size max_workers (lifespan in app.py) with that in
    # mind. Real fix is CAURA-594/595 (hot-path offload).
    #
    # Must stay >= BULK_ENRICHMENT_TOTAL_TIMEOUT_SECONDS in constants.py
    # so the inner cap can actually fire before the outer one.
    request_timeout_seconds: float = 45.0
    # Rate limits applied per-route via slowapi decorators
    # (middleware/rate_limit.py). Syntax: "<count>/<period>" where period
    # is second | minute | hour | day.
    # Mirrors the nginx gateway shape (write_zone 10/s, api_zone 30/s)
    # but keyed by API key rather than IP.
    rate_limit_write: str = "10/second"
    # Bulk write fans out to BULK_MAX_ITEMS=100 memories per request, so a
    # stricter request-level cap keeps the effective memory-write ceiling
    # aligned with the single-write path (2/s * 100 = 200/s vs 10/s single).
    rate_limit_write_bulk: str = "2/second"
    rate_limit_search: str = "30/second"
    # Per-tenant in-flight concurrency caps (see
    # ``middleware/per_tenant_concurrency.py`` for full rationale).
    # Per-instance state — fleet-wide cap is roughly
    # ``cap * max_instances``.
    per_tenant_search_concurrency: int = 8
    per_tenant_write_concurrency: int = 4
    # Fail-fast budget when the cap is exhausted. Long enough to absorb
    # a benign race between two near-simultaneous arrivals; short
    # enough that real exhaustion fails before the request hits the
    # worker.
    per_tenant_acquire_timeout_seconds: float = 0.05
    # Idempotency-Key inbox TTL. 24h matches Stripe's default and is
    # longer than any realistic client retry budget. Cached responses
    # older than this are treated as absent and the request re-runs.
    idempotency_ttl_seconds: int = 86400
    environment: Literal["development", "production", "sandbox"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # JSON output by default so Cloud Logging picks up severity/message.
    # Local developers can set LOG_FORMAT_JSON=false for structlog's
    # coloured ConsoleRenderer.
    log_format_json: bool = True
    # On-prem deployments set this to /var/log/memclaw/core-api/core-api.log so
    # logs land on disk too (daily-rotated, 5-day retention). Empty = stdout only.
    log_file: str = ""
    # Default False: standalone=True bypasses tenant auth, so it must be an explicit opt-in.
    is_standalone: bool = False
    crystallizer_enabled: bool = True
    crystallizer_stale_days: int = 180
    crystallizer_dedup_sample_size: int = 1000
    crystallizer_dedup_threshold: float = 0.95
    core_storage_api_url: str = "http://localhost:8002"
    # Enterprise SaaS splits core-storage-api into writer + reader Cloud Run
    # services (CAURA-591 Part B). When this is set, the storage client
    # routes GET + tagged-read POST calls here instead of ``core_storage_api_url``;
    # empty keeps today's single-service behaviour (OSS + pre-split deploys).
    core_storage_read_url: str = ""
    settings_encryption_key: str = ""  # Required in production (Fernet key)
    jwt_secret: str = "change-me-in-production"  # Required in production
    paddle_client_token: str | None = None
    paddle_environment: str = "sandbox"
    paddle_webhook_secret: str | None = None
    paddle_pro_monthly_price_id: str | None = None
    paddle_pro_annual_price_id: str | None = None
    paddle_business_monthly_price_id: str | None = None
    paddle_business_annual_price_id: str | None = None
    use_stm: bool = False
    stm_backend: str = "memory"  # memory | redis
    stm_notes_ttl: int = 86400  # 24h
    stm_bulletin_ttl: int = 172800  # 48h
    payment_provider: str = "paddle"

    # Platform default providers — Caura's own API keys for tenants without credentials.
    # Set these in enterprise deployments; leave empty for OSS self-hosted.
    platform_llm_provider: str = ""  # "vertex" | "openai" | "" (disabled)
    platform_llm_model: str = ""  # e.g. "gemini-3.1-flash-lite-preview"
    platform_llm_api_key: SecretStr = SecretStr("")  # OpenAI LLM: API key
    platform_llm_gcp_project_id: str = ""  # Vertex: GCP project
    platform_llm_gcp_location: str = ""  # Vertex: region
    platform_embedding_provider: str = ""  # "openai" | "vertex" | "" (disabled)
    platform_embedding_api_key: SecretStr = SecretStr("")  # OpenAI: API key for embeddings
    platform_embedding_model: str = ""  # e.g. "text-embedding-3-small"
    platform_embedding_gcp_project_id: str = (
        ""  # Vertex embedding: GCP project (falls back to platform_llm_gcp_project_id)
    )
    platform_embedding_gcp_location: str = (
        ""  # Vertex embedding: region (falls back to platform_llm_gcp_location)
    )

    # Security audit — scheduler + threshold alerts. Enterprise-only feature;
    # OSS standalone deployments can leave these at defaults (all off).
    # Per-tenant overrides live in tenant_settings.security_audit.
    security_audit_schedule_enabled: bool = False
    security_audit_schedule_cron: str = "0 2 * * *"  # daily 02:00 by default
    security_audit_alerts_enabled: bool = False
    security_audit_alert_recipients: list[str] = []  # comma-separated env → list
    security_audit_alert_score_below: float | None = None
    security_audit_alert_critical_findings_min: int | None = None
    security_audit_alert_score_drop_delta: float | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: Any) -> Any:
        # Pydantic's Literal check rejects invalid values with a clear error
        # after we return — this validator only needs to uppercase so env
        # vars like LOG_LEVEL=debug are accepted.
        return v.upper() if isinstance(v, str) else v

    @field_validator("security_audit_alert_recipients", mode="before")
    @classmethod
    def _split_recipients(cls, v: object) -> object:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @model_validator(mode="after")
    def _validate_timeout_ordering(self) -> "Settings":
        # Local import avoids a circular: constants → common.constants,
        # but this file is imported by constants.py's dependents.
        from core_api.constants import BULK_ENRICHMENT_TOTAL_TIMEOUT_SECONDS

        if self.request_timeout_seconds < BULK_ENRICHMENT_TOTAL_TIMEOUT_SECONDS:
            raise ValueError(
                f"request_timeout_seconds ({self.request_timeout_seconds}s) must be >= "
                f"BULK_ENRICHMENT_TOTAL_TIMEOUT_SECONDS ({BULK_ENRICHMENT_TOTAL_TIMEOUT_SECONDS}s) "
                "so the inner enrichment cap can fire before the outer request budget."
            )
        if self.enrichment_inline_timeout_seconds >= self.request_timeout_seconds:
            raise ValueError(
                f"enrichment_inline_timeout_seconds ({self.enrichment_inline_timeout_seconds}s) "
                f"must be < request_timeout_seconds ({self.request_timeout_seconds}s) so the "
                "inline embed+enrich cap fires before the outer request budget."
            )
        return self

    @field_validator("security_audit_schedule_cron")
    @classmethod
    def _validate_cron_field(cls, v: str) -> str:
        from croniter import CroniterBadCronError, croniter

        try:
            croniter(v)
        except (CroniterBadCronError, ValueError) as exc:
            raise ValueError(f"Invalid cron expression {v!r}: {exc}") from exc
        return v

    @property
    def database_url(self) -> str:
        # Percent-encode the user + password so credentials containing
        # URL-reserved chars (@, :, /, ?, #) or whitespace don't silently
        # produce a malformed URL and an opaque asyncpg connection error.
        # quote(..., safe='') is correct for URL authority components —
        # quote_plus would encode spaces as ``+`` (form-encoding), which
        # is not valid in the userinfo section of a URL.
        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
        )

    @model_validator(mode="after")
    def _remap_deprecated_vertex(self) -> "Settings":
        """Graceful fallback for deprecated tenant-tier ``vertex`` provider.

        Vertex is now platform-tier only. If a deployment still has
        ``EMBEDDING_PROVIDER=vertex`` or ``ENTITY_EXTRACTION_PROVIDER=vertex``
        from a pre-migration env file, remap to ``openai`` at startup rather
        than crashing on the first request through ``get_*_provider``.

        String literals (``"vertex"`` / ``"openai"``) are used here rather
        than ``ProviderName`` members because this validator runs during
        ``settings = Settings()`` at module-bottom import time, and importing
        ``core_api.providers._names`` triggers ``core_api.providers.__init__``
        which imports back from ``core_api.config`` — a circular. StrEnum
        equality means the comparison still works for any future enum-aware
        callers.
        """
        if self.embedding_provider == "vertex":
            logger.warning(
                "EMBEDDING_PROVIDER=vertex is no longer supported as a tenant-facing "
                "provider. Remapping to 'openai'. Configure platform-tier Vertex via "
                "PLATFORM_EMBEDDING_PROVIDER instead."
            )
            object.__setattr__(self, "embedding_provider", "openai")
            # A user coming from Vertex likely has no OPENAI_API_KEY — without a
            # key the registry silently falls back to FakeEmbeddingProvider,
            # which breaks semantic search with no clear signal. Escalate.
            if not self.openai_api_key and not self.platform_embedding_provider:
                logger.error(
                    "EMBEDDING_PROVIDER was remapped from 'vertex' to 'openai', but "
                    "OPENAI_API_KEY is unset and PLATFORM_EMBEDDING_PROVIDER is not "
                    "configured. Semantic search will use FakeEmbeddingProvider and "
                    "produce zero-vectors. Set OPENAI_API_KEY or configure "
                    "PLATFORM_EMBEDDING_PROVIDER to restore embeddings."
                )
        if self.entity_extraction_provider == "vertex":
            logger.warning(
                "ENTITY_EXTRACTION_PROVIDER=vertex is no longer supported as a "
                "tenant-facing provider. Remapping to 'openai'. Configure platform-tier "
                "Vertex via PLATFORM_LLM_PROVIDER instead."
            )
            object.__setattr__(self, "entity_extraction_provider", "openai")
            # Less catastrophic than fake embeddings (LLM enrichment degrades
            # to heuristics) but still worth flagging prominently.
            if not self.openai_api_key and not self.platform_llm_provider:
                logger.error(
                    "ENTITY_EXTRACTION_PROVIDER was remapped from 'vertex' to 'openai', "
                    "but OPENAI_API_KEY is unset and PLATFORM_LLM_PROVIDER is not "
                    "configured. LLM enrichment will use FakeLLMProvider. Set "
                    "OPENAI_API_KEY or configure PLATFORM_LLM_PROVIDER to restore "
                    "enrichment."
                )
        return self

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


def bridge_credentials_to_environ() -> None:
    """Copy ``settings.X`` credential values into ``os.environ``.

    pydantic-settings reads ``.env`` files into the ``Settings``
    instance but does NOT export those values back into
    ``os.environ``. ``common.llm._credentials`` and
    ``common.llm._platform`` (CAURA-595 extraction) read
    ``os.environ`` directly — by design, so core-worker can use them
    without depending on pydantic-settings.

    Without this bridge, a developer with ``OPENAI_API_KEY=sk-...`` in
    ``.env`` (the documented local-dev shape) would silently get
    ``FakeLLMProvider`` for all enrichment / entity-extraction /
    contradiction-detection LLM calls; same for the platform-tier
    singletons configured by ``PLATFORM_*`` settings. The bridge runs
    once during the FastAPI lifespan startup before
    ``init_platform_providers()``.

    Idempotent: only sets keys that aren't already in ``os.environ``,
    so an explicit shell export wins over the ``.env`` value (matches
    pydantic-settings' own precedence: env > ``.env``).
    """
    import os

    bridges: dict[str, str] = {
        # Tenant-tier provider keys read by ``common.llm._credentials``.
        "OPENAI_API_KEY": settings.openai_api_key or "",
        "ANTHROPIC_API_KEY": settings.anthropic_api_key or "",
        "OPENROUTER_API_KEY": settings.openrouter_api_key or "",
        "GEMINI_API_KEY": settings.gemini_api_key or "",
        # Default provider + model used by ``common.enrichment.service``.
        "ENTITY_EXTRACTION_PROVIDER": settings.entity_extraction_provider or "",
        "ENTITY_EXTRACTION_MODEL": settings.entity_extraction_model or "",
        # OpenAI client timeout used by ``common.llm.constants``.
        "OPENAI_REQUEST_TIMEOUT_SECONDS": str(settings.openai_request_timeout_seconds),
        # Platform-tier singletons read by ``common.llm._platform``.
        "PLATFORM_LLM_PROVIDER": settings.platform_llm_provider or "",
        "PLATFORM_LLM_MODEL": settings.platform_llm_model or "",
        "PLATFORM_LLM_API_KEY": (
            settings.platform_llm_api_key.get_secret_value() if settings.platform_llm_api_key else ""
        ),
        "PLATFORM_LLM_GCP_PROJECT_ID": settings.platform_llm_gcp_project_id or "",
        "PLATFORM_LLM_GCP_LOCATION": settings.platform_llm_gcp_location or "",
    }
    for env_name, value in bridges.items():
        if value and not os.environ.get(env_name):
            os.environ[env_name] = value
