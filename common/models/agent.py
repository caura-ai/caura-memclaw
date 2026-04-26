import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, SmallInteger, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from common.models.base import Base


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    fleet_id: Mapped[str | None] = mapped_column(Text)
    agent_id: Mapped[str] = mapped_column(Text, nullable=False)
    trust_level: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("1"))
    search_profile: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("uq_agents_tenant_agent", "tenant_id", "agent_id", unique=True),
    )
