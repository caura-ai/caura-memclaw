"""Service configuration — env vars validated at startup."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "production", "sandbox"] = "development"

    # Logging
    log_level: str = "INFO"
    log_format_json: bool = False
    log_file: str = ""

    # Storage backend — the worker PATCHes embeddings to core-storage-api
    # via this URL. Defaults to the local docker-compose service name.
    core_storage_api_url: str = "http://oss-core-storage-api:8002"

    # Event bus — `inprocess` for tests/standalone OSS, `pubsub` for SaaS.
    event_bus_backend: Literal["inprocess", "pubsub"] = "inprocess"
    gcp_project_id: str = "local-dev"
    # One subscription per consumer service so a single topic can fan
    # out to multiple distinct subscribers; this prefix uniquely names
    # the worker's subscription on shared topics.
    event_bus_subscription_prefix: str = "core-worker"

    # HTTP timeout for the storage PATCH. The worker is off the request
    # hot path so a longer timeout is fine; we'd rather wait + succeed
    # than 504 + nack + redeliver.
    storage_http_timeout_s: float = 30.0

    # Cloud Run port.
    port: int = 8080
