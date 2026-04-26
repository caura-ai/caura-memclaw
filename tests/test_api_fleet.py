"""E2E fleet management tests through HTTP API — real DB, no mocks."""

import uuid

import pytest

from tests.conftest import get_test_auth, uid as _uid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _heartbeat(client, tenant_id: str, headers: dict, node_name: str,
                     fleet_id: str, agents: list | None = None,
                     version: str = "1.0.0", platform: str = "linux") -> dict:
    """Send a heartbeat and return the response JSON."""
    resp = await client.post(
        "/api/v1/fleet/heartbeat",
        json={
            "tenant_id": tenant_id,
            "fleet_id": fleet_id,
            "node_name": node_name,
            "openclaw_version": version,
            "os_info": platform,
            "agents": agents or [],
        },
        headers=headers,
    )
    assert resp.status_code == 200, f"Heartbeat failed: {resp.text}"
    return resp.json()


async def _get_nodes(client, tenant_id: str, headers: dict) -> list:
    resp = await client.get(
        f"/api/v1/fleet/nodes?tenant_id={tenant_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_heartbeat_registers_node(client):
    """POST /api/fleet/heartbeat registers a new node, GET /api/fleet/nodes returns it."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fid = f"fleet-{tag}"

    hb = await _heartbeat(client, tenant_id, headers, f"node-alpha-{tag}", fid)
    assert hb["ok"] is True
    assert "node_id" in hb

    nodes = await _get_nodes(client, tenant_id, headers)
    names = [n["node_name"] for n in nodes]
    assert f"node-alpha-{tag}" in names

    node = next(n for n in nodes if n["node_name"] == f"node-alpha-{tag}")
    assert node["fleet_id"] == fid
    assert node["status"] == "online"


async def test_heartbeat_updates_existing_node(client):
    """Two heartbeats with the same node_name produce only one node row."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fid = f"fleet-{tag}"

    await _heartbeat(client, tenant_id, headers, f"node-beta-{tag}", fid, version="1.0.0")
    await _heartbeat(client, tenant_id, headers, f"node-beta-{tag}", fid, version="2.0.0")

    nodes = await _get_nodes(client, tenant_id, headers)
    beta_nodes = [n for n in nodes if n["node_name"] == f"node-beta-{tag}"]
    assert len(beta_nodes) == 1
    assert beta_nodes[0]["openclaw_version"] == "2.0.0"


async def test_list_fleets(client):
    """Heartbeats to 2 fleet_ids → GET /api/fleets returns both with correct agent counts."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fid_a = f"fleet-a-{tag}"
    fid_b = f"fleet-b-{tag}"

    # Write memories with fleet_ids so /api/fleets (which queries memories) can see them
    for i, fid in enumerate([fid_a, fid_b]):
        for j in range(2):
            await client.post(
                "/api/v1/memories",
                json={
                    "tenant_id": tenant_id,
                    "content": f"Fleet {fid} memory {j} [{tag}]",
                    "agent_id": f"agent-{tag}-{i}-{j}",
                    "fleet_id": fid,
                    "memory_type": "fact",
                },
                headers=headers,
            )

    resp = await client.get(
        f"/api/v1/fleets?tenant_id={tenant_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    fleets = resp.json()
    fleet_ids = [f["fleet_id"] for f in fleets]
    assert fid_a in fleet_ids
    assert fid_b in fleet_ids

    for f in fleets:
        if f["fleet_id"] in (fid_a, fid_b):
            assert f["memory_count"] == 2
            assert f["agent_count"] == 2


async def test_dispatch_command(client):
    """Register node, POST command, GET commands → command appears."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fid = f"fleet-{tag}"

    hb = await _heartbeat(client, tenant_id, headers, f"node-cmd-{tag}", fid)
    node_id = hb["node_id"]

    # Dispatch a command
    resp = await client.post(
        "/api/v1/fleet/commands",
        json={
            "node_id": node_id,
            "command": "ping",
            "payload": {"msg": "hello"},
        },
        headers=headers,
    )
    assert resp.status_code == 201
    cmd_data = resp.json()
    assert cmd_data["status"] == "pending"

    # List commands
    resp = await client.get(
        f"/api/v1/fleet/commands?tenant_id={tenant_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    commands = resp.json()
    assert any(c["command"] == "ping" and c["node_id"] == node_id for c in commands)


async def test_node_agents_from_heartbeat(client):
    """Heartbeat with agents list → GET nodes returns node with those agents."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fid = f"fleet-{tag}"

    agents = [f"agent-x-{tag}", f"agent-y-{tag}", f"agent-z-{tag}"]
    await _heartbeat(client, tenant_id, headers, f"node-agents-{tag}", fid, agents=agents)

    nodes = await _get_nodes(client, tenant_id, headers)
    node = next(n for n in nodes if n["node_name"] == f"node-agents-{tag}")
    assert node["agents"] == agents
    assert len(node["agents"]) == 3
