"""Database initialization — runs Alembic migrations on startup."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from core_storage_api.config import settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_read_engine: AsyncEngine | None = None


def _build_engine(url: str) -> AsyncEngine:
    return create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
    )


def get_engine() -> AsyncEngine:
    """Return the writer engine (primary DB), creating on first call."""
    global _engine
    if _engine is None:
        _engine = _build_engine(settings.database_url)
    return _engine


def get_read_engine() -> AsyncEngine:
    """Return the reader engine. Same as writer when ``read_database_url``
    isn't configured (OSS standalone); otherwise its own pool against
    the replica so read traffic doesn't share primary's connection
    budget."""
    global _read_engine
    if not settings.read_database_url:
        return get_engine()
    if _read_engine is None:
        _read_engine = _build_engine(settings.read_database_url)
    return _read_engine


async def get_session():
    """Yield an async session for writes / read-after-write work."""
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(get_engine(), expire_on_commit=False) as session:
        yield session


async def init_database() -> None:
    """Run all pending Alembic migrations to initialize/update the database schema.

    If tables already exist (e.g., created by the legacy backend), stamps the
    current revision so Alembic skips the initial migration.

    Uses a PostgreSQL advisory lock so that when multiple uvicorn workers start
    concurrently, only one runs migrations; the others wait and then no-op.

    Role=reader (CAURA-591 Part B): no-op. The writer owns schema; reader-role
    services connect to the read-replica pool which rejects DDL anyway, so
    attempting to migrate would just fail with a confusing error.
    """
    if settings.core_storage_role == "reader":
        logger.info("Skipping Alembic (role=reader — writer owns schema)")
        return

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    engine = get_engine()
    migrations_dir = str(Path(__file__).parent / "migrations")

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", migrations_dir)

    # Acquire a transaction-level advisory lock so concurrent workers serialise.
    # pg_advisory_xact_lock is released on transaction commit or rollback,
    # which is correct for a pooled connection where the session never closes.
    _MIGRATION_LOCK_ID = 8_675_309  # arbitrary unique int
    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL lock_timeout = '120s'"))
        await conn.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _MIGRATION_LOCK_ID},
        )

        has_tables = await conn.scalar(
            text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'memories')"
            )
        )
        has_alembic = await conn.scalar(
            text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'alembic_version')"
            )
        )

        def _run_upgrade(connection: Connection) -> None:
            alembic_cfg.attributes["connection"] = connection
            if has_tables and not has_alembic:
                # Tables exist from legacy backend — stamp as current, skip creation
                logger.info("Existing tables detected, stamping Alembic at head")
                command.stamp(alembic_cfg, "head")
            else:
                command.upgrade(alembic_cfg, "head")

        await conn.run_sync(_run_upgrade)

    logger.info("Database initialization complete")
