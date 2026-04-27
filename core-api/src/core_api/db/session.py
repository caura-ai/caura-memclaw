import contextvars
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core_api.config import settings

# asyncpg: enable SSL when the database requires it (default for managed
# Postgres providers — required, e.g., on AlloyDB's ENCRYPTED_ONLY mode).
_connect_args = {}
if settings.postgres_require_ssl:
    _connect_args["ssl"] = "require"
    if settings.postgres_use_iam_auth:
        _connect_args["ssl"] = True  # full verification for IAM auth

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=int(getattr(settings, "db_pool_size", 50)),
    max_overflow=int(getattr(settings, "db_max_overflow", 50)),
    pool_timeout=120,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args=_connect_args,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Context variable for RLS: set by auth middleware before DB access
_current_tenant_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_tenant_id", default=None
)


def set_current_tenant(tenant_id: str | None) -> None:
    """Set the tenant_id for RLS enforcement in the current request context."""
    _current_tenant_id.set(tenant_id)


def get_current_tenant() -> str | None:
    """Get the tenant_id for the current request context."""
    return _current_tenant_id.get()


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with async_session() as session:
        # Set RLS context variable for this session
        tenant_id = _current_tenant_id.get()
        if tenant_id:
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": tenant_id},
            )
        else:
            await session.execute(text("SELECT set_config('app.tenant_id', '__admin__', true)"))
        yield session
