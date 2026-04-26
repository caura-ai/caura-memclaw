"""E2E public stats endpoint tests through HTTP API."""

import pytest

from tests.conftest import get_test_auth, uid as _uid


# ---------------------------------------------------------------------------
# GET /api/stats — public counters
# ---------------------------------------------------------------------------


async def test_stats_returns_counts(client):
    """GET /api/stats returns tenant_count and memory_count."""
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "tenant_count" in data
    assert "memory_count" in data
    assert isinstance(data["tenant_count"], int)
    assert isinstance(data["memory_count"], int)
    assert data["tenant_count"] >= 1  # standalone mode always has 1 tenant


async def test_stats_reflect_actual_data(client):
    """After writing a memory, memory_count should increase."""
    tenant_id, headers = get_test_auth()

    # Get baseline
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    before = resp.json()["memory_count"]

    # Write a memory
    tag = _uid()
    resp = await client.post(
        "/api/v1/memories",
        json={
            "tenant_id": tenant_id,
            "content": f"Stats counter test [{tag}]",
            "agent_id": f"stats-agent-{tag}",
            "fleet_id": f"stats-fleet-{tag}",
            "memory_type": "fact",
        },
        headers=headers,
    )
    assert resp.status_code == 201

    # Verify count increased
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    after = resp.json()["memory_count"]
    assert after >= before + 1


async def test_stats_no_auth_required(client):
    """GET /api/stats works without any auth headers (public endpoint)."""
    resp = await client.get("/api/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "tenant_count" in data
    assert "memory_count" in data
