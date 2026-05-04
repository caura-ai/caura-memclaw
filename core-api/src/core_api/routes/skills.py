"""Shared-skill distribution REST surface.

Mirrors the MCP ``memclaw_share_skill`` / ``memclaw_unshare_skill``
tools — both call into ``services.skill_service``. Skills are stored
as documents in the ``skills`` collection (with the ``description``
field auto-embedded for semantic search) and distributed to plugin
nodes via the existing ``fleet_commands`` queue.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core_api.auth import AuthContext, get_auth_context
from core_api.db.session import get_db
from core_api.services.audit_service import log_action
from core_api.services.skill_service import list_skills, share_skill, unshare_skill

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Skills"])


class SkillShareRequest(BaseModel):
    tenant_id: str | None = None
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    target_fleet_id: str = Field(min_length=1)
    install_on_fleet: bool = Field(
        default=False,
        description=(
            "If true, queue install_skill fleet commands so every node in "
            "target_fleet_id auto-installs the skill on next heartbeat. "
            "If false (default), only publish to the catalog — recipients "
            "discover via memclaw_doc op=query collection=skills."
        ),
    )
    author_agent_id: str | None = None
    target_agent_ids: list[str] | None = None
    version: int = Field(default=1, ge=1)


class SkillShareResponse(BaseModel):
    skill_id: str
    name: str
    target_fleet_id: str
    install_on_fleet: bool
    queued_nodes: int
    node_ids: list[str]


class SkillSummary(BaseModel):
    skill_id: str
    name: str
    description: str
    author_agent_id: str | None
    target_fleet_id: str | None
    target_agent_ids: list[str]
    version: int
    similarity: float | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SkillUnshareResponse(BaseModel):
    name: str
    deleted: bool
    unshare_from_fleet: bool
    target_fleet_id: str | None
    queued_nodes: int
    node_ids: list[str]


@router.post("/skills/share", response_model=SkillShareResponse)
async def share_skill_endpoint(
    body: SkillShareRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SkillShareResponse:
    """Share a skill with a fleet.

    Default mode upserts the skill into the ``skills`` document
    collection (with the ``description`` field embedded for semantic
    search) — recipients pull on demand. Pass ``install_on_fleet=true``
    to also queue ``install_skill`` fleet commands per node.
    """
    tenant_id = body.tenant_id or auth.tenant_id
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")
    auth.enforce_tenant(tenant_id)
    auth.enforce_read_only()

    try:
        result = await share_skill(
            db=db,
            tenant_id=tenant_id,
            name=body.name,
            description=body.description,
            content=body.content,
            target_fleet_id=body.target_fleet_id,
            install_on_fleet=body.install_on_fleet,
            author_agent_id=body.author_agent_id,
            target_agent_ids=body.target_agent_ids,
            version=body.version,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    await log_action(
        db,
        tenant_id=tenant_id,
        action="skill_share",
        resource_type="skill",
        resource_id=result["skill_id"],
        detail={
            "name": result["name"],
            "target_fleet_id": result["target_fleet_id"],
            "install_on_fleet": result["install_on_fleet"],
            "queued_nodes": result["queued_nodes"],
        },
    )
    await db.commit()

    return SkillShareResponse(
        skill_id=result["skill_id"],
        name=result["name"],
        target_fleet_id=result["target_fleet_id"],
        install_on_fleet=result["install_on_fleet"],
        queued_nodes=result["queued_nodes"],
        node_ids=result["node_ids"],
    )


@router.delete("/skills/{name}", response_model=SkillUnshareResponse)
async def unshare_skill_endpoint(
    name: str,
    tenant_id: str | None = Query(default=None),
    target_fleet_id: str | None = Query(
        default=None,
        description="Required when unshare_from_fleet=true.",
    ),
    unshare_from_fleet: bool = Query(
        default=False,
        description=(
            "If true, queue uninstall_skill fleet commands so every node in "
            "target_fleet_id deletes its local SKILL.md. If false (default), "
            "only remove from the catalog — already-installed nodes keep "
            "their copy."
        ),
    ),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> SkillUnshareResponse:
    """Remove a skill from the catalog (and optionally from fleet nodes)."""
    if tenant_id:
        auth.enforce_tenant(tenant_id)
    elif not auth.is_admin:
        if not auth.tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required")
        tenant_id = auth.tenant_id
    auth.enforce_read_only()

    try:
        result = await unshare_skill(
            db=db,
            tenant_id=tenant_id or "",
            name=name,
            unshare_from_fleet=unshare_from_fleet,
            target_fleet_id=target_fleet_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    await log_action(
        db,
        tenant_id=tenant_id or "",
        action="skill_unshare",
        resource_type="skill",
        detail={
            "name": result["name"],
            "deleted": result["deleted"],
            "unshare_from_fleet": result["unshare_from_fleet"],
            "queued_nodes": result["queued_nodes"],
        },
    )
    await db.commit()

    return SkillUnshareResponse(**result)


@router.get("/skills", response_model=list[SkillSummary])
async def list_skills_endpoint(
    tenant_id: str | None = Query(default=None),
    fleet_id: str | None = Query(default=None),
    query: str | None = Query(
        default=None,
        description=(
            "Natural-language query — if provided, runs semantic search "
            "over the embedded skill descriptions and returns top-``limit`` "
            "matches by cosine similarity. Omit to list by recency."
        ),
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
) -> list[SkillSummary]:
    """List or semantically search skills available to the caller."""
    if tenant_id:
        auth.enforce_tenant(tenant_id)
    elif not auth.is_admin:
        if not auth.tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required")
        tenant_id = auth.tenant_id

    summaries = await list_skills(
        db=db,
        tenant_id=tenant_id or "",
        fleet_id=fleet_id,
        query=query,
        limit=limit,
        offset=offset,
    )
    return [SkillSummary(**s) for s in summaries]
