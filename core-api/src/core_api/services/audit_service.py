from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core_api.clients.storage_client import get_storage_client


async def log_action(
    db: AsyncSession,
    *,
    tenant_id: str,
    agent_id: str | None = None,
    action: str,
    resource_type: str,
    resource_id: UUID | None = None,
    detail: dict | None = None,
) -> None:
    sc = get_storage_client()
    await sc.create_audit_log(
        {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": str(resource_id) if resource_id else None,
            "detail": detail,
        }
    )
    # Note: the storage API handles persistence; no separate commit needed.
