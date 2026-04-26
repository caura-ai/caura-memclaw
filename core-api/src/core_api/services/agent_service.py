"""Agent trust-level enforcement for fleet-scoped access control."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core_api.clients.storage_client import get_storage_client
from core_api.constants import DEFAULT_TRUST_LEVEL
from core_api.services.audit_service import log_action

logger = logging.getLogger(__name__)


async def get_or_create_agent(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    fleet_id: str | None = None,
    *,
    require_approval: bool = False,
) -> dict:
    """Return the agent dict, creating it on first encounter.

    The storage API handles upsert semantics and race-condition safety.
    """
    sc = get_storage_client()
    agent = await sc.get_agent(agent_id, tenant_id)
    if agent:
        # Backfill fleet_id if the agent was registered without one
        if agent.get("fleet_id") is None and fleet_id is not None:
            agent["fleet_id"] = fleet_id
            agent["updated_at"] = datetime.now(UTC)
            sc = get_storage_client()
            await sc.create_or_update_agent(
                {
                    "tenant_id": tenant_id,
                    "agent_id": agent_id,
                    "fleet_id": fleet_id,
                }
            )
        return agent

    initial_trust = 0 if require_approval else DEFAULT_TRUST_LEVEL
    agent = await sc.create_or_update_agent(
        {
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "fleet_id": fleet_id,
            "trust_level": initial_trust,
        }
    )
    await log_action(
        db,
        tenant_id=tenant_id,
        agent_id=agent_id,
        action="agent_registered",
        resource_type="agent",
        resource_id=agent.get("id"),
        detail={"fleet_id": fleet_id, "trust_level": initial_trust},
    )
    return agent


async def lookup_agent(db: AsyncSession, tenant_id: str, agent_id: str) -> dict | None:
    sc = get_storage_client()
    return await sc.get_agent(agent_id, tenant_id)


async def enforce_fleet_write(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    fleet_id: str | None,
) -> dict:
    """Enforce write permissions. Returns the agent (auto-created if new)."""
    agent = await get_or_create_agent(db, tenant_id, agent_id, fleet_id)

    # Agents can always write to their home fleet (or tenant-wide if no fleet specified)
    if fleet_id is None or fleet_id == agent.get("fleet_id"):
        return agent

    # Cross-fleet write requires admin (level >= 3)
    trust = agent.get("trust_level", 0)
    if trust < 3:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' (trust_level={trust}) cannot write to fleet '{fleet_id}'. Requires trust_level >= 3.",
        )
    return agent


async def enforce_fleet_read(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    fleet_id: str | None,
) -> None:
    """Enforce read permissions for search/list (read-only — never creates agents)."""
    agent = await lookup_agent(db, tenant_id, agent_id)

    # Unknown agent — allow the read (agent registration happens on writes)
    if not agent:
        return

    # Reading own fleet or tenant-wide is always allowed
    if fleet_id is None or fleet_id == agent.get("fleet_id"):
        return

    # Cross-fleet read requires level >= 2
    trust = agent.get("trust_level", 0)
    if trust < 2:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' (trust_level={trust}) cannot read fleet '{fleet_id}'. Requires trust_level >= 2.",
        )


async def enforce_delete(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
) -> None:
    """Enforce delete permissions."""
    agent = await lookup_agent(db, tenant_id, agent_id)
    if not agent:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' is not registered and cannot delete memories.",
        )

    trust = agent.get("trust_level", 0)
    if trust < 3:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' (trust_level={trust}) cannot delete memories. Requires trust_level >= 3.",
        )


async def enforce_update(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    memory_owner_agent_id: str,
) -> None:
    """Enforce update permissions. Level 0-2 can only update own memories; level 3 can update any."""
    agent = await lookup_agent(db, tenant_id, agent_id)
    if not agent:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' is not registered and cannot update memories.",
        )
    trust = agent.get("trust_level", 0)
    if trust == 0:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' (trust_level=0) is restricted from updates.",
        )
    if trust < 3 and agent_id != memory_owner_agent_id:
        raise HTTPException(
            status_code=403,
            detail=f"Agent '{agent_id}' (trust_level={trust}) can only update own memories. Requires trust_level >= 3.",
        )


async def backfill_agents(db: AsyncSession) -> int:
    """Create agent rows for any (tenant_id, agent_id) pairs in memories that don't have one yet."""
    sc = get_storage_client()
    # Use the first available tenant_id — in standalone mode there's only one
    from core_api.standalone import get_standalone_tenant_id

    tenant_id = get_standalone_tenant_id()
    result = await sc.backfill_from_memories(tenant_id)
    return result.get("count", 0)


async def update_trust_level(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    trust_level: int,
    fleet_id: str | None = None,
) -> dict:
    """Update an agent's trust level (and optionally fleet). Returns the updated agent."""
    agent = await lookup_agent(db, tenant_id, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    sc = get_storage_client()
    data: dict[str, Any] = {"tenant_id": tenant_id, "trust_level": trust_level}
    if fleet_id is not None:
        data["fleet_id"] = fleet_id
    await sc.update_trust_level(agent_id, data)
    # Re-fetch to get the updated agent dict
    updated = await sc.get_agent(agent_id, tenant_id)
    return updated or agent
