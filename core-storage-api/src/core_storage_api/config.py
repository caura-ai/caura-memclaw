"""Service configuration — all env vars validated at startup."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    environment: Literal["development", "production", "sandbox"] = "development"

    # Database — writes go to the primary at ``database_url``. Reads
    # route to ``read_database_url`` when it's set (SaaS wires this at
    # the AlloyDB read pool; the IP lives in the enterprise deploy
    # config, not here) and fall back to the primary when empty (OSS
    # standalone — a single box with no replica). The split offloads
    # search / GET traffic from the primary and unlocks the existing
    # but idle read pool. Replication lag on AlloyDB is typically
    # <5s, acceptable for the read paths we route.
    database_url: str = "postgresql+asyncpg://memclaw:changeme@localhost:5432/memclaw"
    read_database_url: str = ""
    db_pool_size: int = 20
    db_max_overflow: int = 20
    db_pool_timeout: int = 60
    db_pool_recycle: int = 1800

    # Service role (CAURA-591 Part B). "hybrid" keeps the original
    # single-service behaviour and is the safe default for OSS + any
    # deploy that hasn't opted into the split. Enterprise SaaS runs
    # two Cloud Run services: the writer (role=writer) owns schema +
    # serves POST/PATCH/DELETE, the reader (role=reader) runs no
    # migrations, skips write routes, and uses the read-pool URL as
    # its primary connection.
    core_storage_role: Literal["writer", "reader", "hybrid"] = "hybrid"

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # JSON output by default so Cloud Logging picks up severity/message.
    # Local developers can set LOG_FORMAT_JSON=false for structlog's
    # coloured ConsoleRenderer.
    log_format_json: bool = True
    # On-prem deployments set this to /var/log/memclaw/<service>/<service>.log
    # so logs land on disk too (daily-rotated, 5-day retention). Empty string
    # means stdout only — the SaaS default, unchanged.
    log_file: str = ""

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> Any:
        # Pydantic's Literal check rejects invalid values with a clear error
        # after we return — this validator only needs to uppercase so env
        # vars like LOG_LEVEL=debug are accepted.
        return v.upper() if isinstance(v, str) else v

    # CORS — internal service, restrict to known callers
    cors_origins: str = "http://localhost:8000"

    # Server
    host: str = "0.0.0.0"
    port: int = 8002

    # Scoring
    # Soft boost applied to memories whose anchor date falls inside the
    # query-extracted date range.  Replaces the old hard WHERE filter so
    # semantically strong out-of-range memories stay retrievable.
    date_range_boost_factor: float = 2.0

    # Soft penalty applied to memories whose ts_valid_end is in the past
    # relative to the query's valid_at.  Replaces the old hard WHERE filter
    # (`ts_valid_end >= valid_at`) — an over-eager enrichment date no longer
    # catastrophically hides the memory, just down-weights it.
    expired_currency_factor: float = 0.5


settings = Settings()  # type: ignore[call-arg]
