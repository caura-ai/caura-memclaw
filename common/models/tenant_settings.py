"""Per-tenant settings storage: live overrides + append-only audit trail."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.models.base import Base


class TenantSettings(Base):
    """Live tenant overrides — one row per tenant, JSONB holds only overrides."""

    __tablename__ = "tenant_settings"

    tenant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    settings: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class TenantSettingsAudit(Base):
    """Append-only per-change audit trail. One row per PUT /settings that changes a value."""

    __tablename__ = "tenant_settings_audit"
    __table_args__ = (
        Index(
            "idx_tenant_settings_audit_tenant_created",
            "tenant_id",
            text("created_at DESC"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    changed_by: Mapped[str | None] = mapped_column(Text)
    diff: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
