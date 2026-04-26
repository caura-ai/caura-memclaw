"""Fleet heartbeat and command channel — replaces WebSocket/SSH gateway model."""

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core_api.auth import AuthContext, get_auth_context
from core_api.clients.storage_client import get_storage_client
from core_api.constants import NODE_OFFLINE_SECONDS, NODE_STALE_SECONDS
from core_api.db.session import get_db
from core_api.services.audit_service import log_action

router = APIRouter(tags=["Fleet"])


# ── Schemas ──


class FleetCreateIn(BaseModel):
    tenant_id: str
    fleet_id: str  # alphanumeric + hyphens, 3-50 chars
    display_name: str | None = None
    description: str | None = None

    @classmethod
    def validate_fleet_id(cls, v: str) -> str:
        import re

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-]{1,48}[a-zA-Z0-9]$", v):
            raise ValueError(
                "fleet_id must be 3-50 chars, alphanumeric + hyphens, no leading/trailing hyphens"
            )
        return v


class HeartbeatIn(BaseModel):
    tenant_id: str
    node_name: str
    fleet_id: str | None = None
    hostname: str | None = None
    ip: str | None = None
    openclaw_version: str | None = None
    plugin_version: str | None = None
    plugin_hash: str | None = None
    os_info: str | None = None
    agents: list | None = None
    tools: list | None = None
    channels: list | None = None
    metadata: dict | None = None


class CommandIn(BaseModel):
    tenant_id: str | None = None
    node_id: UUID
    command: str
    payload: dict | None = None


class CommandResultIn(BaseModel):
    status: str  # done | failed
    result: dict | None = None


# ── Fleet CRUD ──


@router.post("/fleet", status_code=201)
async def create_fleet(
    body: FleetCreateIn,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """Explicitly create a fleet (team) within a tenant."""
    auth.enforce_read_only()
    auth.enforce_usage_limits()
    auth.enforce_tenant(body.tenant_id)

    # Validate fleet_id format
    FleetCreateIn.validate_fleet_id(body.fleet_id)

    sc = get_storage_client()

    # Check if fleet already exists
    if await sc.fleet_exists(body.tenant_id, body.fleet_id):
        raise HTTPException(status_code=409, detail=f"Fleet '{body.fleet_id}' already exists")

    # Create a sentinel node to register the fleet
    await sc.upsert_node(
        {
            "tenant_id": body.tenant_id,
            "node_name": f"_fleet_{body.fleet_id}",
            "fleet_id": body.fleet_id,
            "metadata": {
                "display_name": body.display_name,
                "description": body.description,
                "sentinel": True,
            },
            "last_heartbeat": datetime.now(UTC).isoformat(),
        }
    )
    await log_action(
        db,
        tenant_id=body.tenant_id,
        action="create",
        resource_type="fleet",
        detail={"fleet_id": body.fleet_id, "display_name": body.display_name},
    )
    await db.commit()
    return {"ok": True, "fleet_id": body.fleet_id, "tenant_id": body.tenant_id}


@router.get("/fleet")
async def list_fleets(
    tenant_id: str = Query(...),
    auth: AuthContext = Depends(get_auth_context),
):
    """List distinct fleets for a tenant with node counts."""
    auth.enforce_tenant(tenant_id)

    sc = get_storage_client()
    rows = await sc.list_fleets(tenant_id)
    now = datetime.now(UTC)
    return [
        {
            "fleet_id": r.get("fleet_id"),
            "node_count": int(r.get("node_count", 0)),
            "last_heartbeat": r.get("last_heartbeat"),
            "status": "online"
            if r.get("last_heartbeat") and _age_seconds(r.get("last_heartbeat"), now) < NODE_OFFLINE_SECONDS
            else "offline",
        }
        for r in rows
    ]


@router.delete("/fleet/{fleet_id}", status_code=204)
async def delete_fleet(
    fleet_id: str,
    tenant_id: str = Query(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """Delete a fleet and all its nodes. Memories are NOT deleted (they retain fleet_id for history)."""
    auth.enforce_read_only()
    auth.enforce_tenant(tenant_id)

    sc = get_storage_client()

    # Count nodes to delete
    node_count = await sc.count_nodes(tenant_id=tenant_id, fleet_id=fleet_id)
    if node_count == 0:
        raise HTTPException(status_code=404, detail=f"Fleet '{fleet_id}' not found")

    # Delete all commands for fleet nodes, then delete nodes
    await sc.delete_fleet(tenant_id=tenant_id, fleet_id=fleet_id)
    await log_action(
        db,
        tenant_id=tenant_id,
        action="delete",
        resource_type="fleet",
        detail={"fleet_id": fleet_id, "nodes_deleted": node_count},
    )
    await db.commit()


# ── Heartbeat ──


@router.post("/fleet/heartbeat")
async def heartbeat(
    body: HeartbeatIn,
    auth: AuthContext = Depends(get_auth_context),
):
    """Plugin pushes status; receives pending commands in response."""
    auth.enforce_tenant(body.tenant_id)

    now = datetime.now(UTC)
    sc = get_storage_client()
    node = await sc.upsert_node(
        {
            "tenant_id": body.tenant_id,
            "node_name": body.node_name,
            "fleet_id": body.fleet_id,
            "hostname": body.hostname,
            "ip": body.ip,
            "openclaw_version": body.openclaw_version,
            "plugin_version": body.plugin_version,
            "plugin_hash": body.plugin_hash,
            "os_info": body.os_info,
            "agents_json": body.agents,
            "tools_json": body.tools,
            "channels_json": body.channels,
            "metadata": body.metadata,
            "last_heartbeat": now.isoformat(),
        }
    )

    node_id = node.get("id", "")
    node_name = node.get("node_name", body.node_name)

    # Fetch pending commands and mark as acked
    commands = await sc.get_pending_commands(body.tenant_id, node_name)
    if commands:
        await sc.ack_commands([c.get("id") for c in commands])

    return {
        "ok": True,
        "node_id": str(node_id),
        "commands": [
            {
                "id": str(c.get("id", "")),
                "command": c.get("command"),
                "payload": c.get("payload"),
            }
            for c in commands
        ],
    }


# ── Command result ──


@router.post("/fleet/commands/{command_id}/result")
async def command_result(
    command_id: UUID,
    body: CommandResultIn,
    auth: AuthContext = Depends(get_auth_context),
):
    """Plugin reports command completion."""
    sc = get_storage_client()
    updated = await sc.update_command_status(
        str(command_id),
        {
            "status": body.status,
            "result": body.result,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Command not found")
    return {"ok": True}


# ── Fleet nodes (frontend reads) ──


@router.get("/fleet/nodes")
async def list_nodes(
    tenant_id: str = Query(...),
    fleet_id: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    """List fleet nodes for a tenant with computed status."""
    auth.enforce_tenant(tenant_id)

    sc = get_storage_client()
    nodes = await sc.list_nodes(tenant_id, fleet_id=fleet_id)
    now = datetime.now(UTC)

    out = []
    for n in nodes:
        hb = n.get("last_heartbeat")
        age = _age_seconds(hb, now) if hb else 999999
        if age > NODE_OFFLINE_SECONDS:
            status = "offline"
        elif age > NODE_STALE_SECONDS:
            status = "stale"
        else:
            status = "online"

        out.append(
            {
                "node_id": str(n.get("id", "")),
                "node_name": n.get("node_name"),
                "fleet_id": n.get("fleet_id"),
                "hostname": n.get("hostname"),
                "ip": n.get("ip"),
                "openclaw_version": n.get("openclaw_version"),
                "plugin_version": n.get("plugin_version"),
                "plugin_hash": n.get("plugin_hash"),
                "os_info": n.get("os_info"),
                "status": status,
                "agents": n.get("agents_json"),
                "tools": n.get("tools_json"),
                "channels": n.get("channels_json"),
                "metadata": n.get("metadata"),
                "last_heartbeat": n.get("last_heartbeat"),
                "created_at": n.get("created_at"),
            }
        )

    return out


# ── Fleet & agent stats ──


@router.get("/fleet/stats")
async def fleet_stats(
    tenant_id: str = Query(...),
    fleet_id: str | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    """Per-agent and fleet-level memory stats for the Fleet UI."""
    auth.enforce_tenant(tenant_id)
    sc = get_storage_client()
    return await sc.fleet_stats(tenant_id, fleet_id)


# ── Queue command (frontend posts) ──


@router.post("/fleet/commands", status_code=201)
async def create_command(
    body: CommandIn,
    auth: AuthContext = Depends(get_auth_context),
):
    """Queue a command for a fleet node."""
    sc = get_storage_client()
    # We need to verify node exists and get its tenant_id
    # The storage client get_node takes tenant_id + node_name, but we have node_id
    # Create the command via the storage client
    tenant_id = body.tenant_id or auth.tenant_id
    cmd = await sc.create_command(
        {
            "tenant_id": tenant_id,
            "node_id": str(body.node_id),
            "command": body.command,
            "payload": body.payload,
        }
    )
    return {"id": str(cmd.get("id", "")), "status": cmd.get("status", "pending")}


# ── Command history ──


@router.get("/fleet/commands")
async def list_commands(
    tenant_id: str = Query(...),
    node_id: UUID | None = Query(default=None),
    auth: AuthContext = Depends(get_auth_context),
):
    """List recent commands for a tenant, optionally filtered by node."""
    auth.enforce_tenant(tenant_id)

    sc = get_storage_client()
    commands = await sc.list_commands(tenant_id=tenant_id)

    return [
        {
            "id": str(c.get("id", "")),
            "node_id": str(c.get("node_id", "")),
            "command": c.get("command"),
            "payload": c.get("payload"),
            "status": c.get("status"),
            "result": c.get("result"),
            "created_at": c.get("created_at"),
            "acked_at": c.get("acked_at"),
            "completed_at": c.get("completed_at"),
        }
        for c in commands
    ]


# ── Helpers ──


def _age_seconds(timestamp: str | None, now: datetime) -> float:
    """Compute age in seconds from an ISO timestamp string."""
    if not timestamp:
        return 999999
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (now - dt).total_seconds()
    except (ValueError, TypeError):
        return 999999
