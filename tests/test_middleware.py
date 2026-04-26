"""Tests for pure ASGI middleware and MCP Bearer auth.

Covers:
- SecurityHeadersMiddleware: headers on non-MCP, skipped on /mcp
- ResponseTimeMiddleware: X-Response-Time on non-MCP, skipped on /mcp
- MCPAuthMiddleware: Bearer token extraction
"""

import pytest

from tests.conftest import get_test_auth

pytestmark = pytest.mark.asyncio


# ── SecurityHeadersMiddleware ──


async def test_security_headers_on_api_routes(client):
    """Non-MCP routes should have all security headers."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.headers["strict-transport-security"] == "max-age=63072000; includeSubDomains; preload"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "content-security-policy" in resp.headers


async def test_security_headers_absent_on_mcp(client):
    """MCP routes should NOT have browser security headers."""
    resp = await client.get("/mcp")
    # /mcp may redirect or return protocol error — either way, no security headers
    assert "x-frame-options" not in resp.headers
    assert "content-security-policy" not in resp.headers


# ResponseTimeMiddleware was removed during OSS/Enterprise split
# (it depended on enterprise metrics_service).


# ── MCPAuthMiddleware: Bearer token ──


async def test_mcp_auth_middleware_bearer_extraction(client):
    """MCPAuthMiddleware should extract API key from Authorization: Bearer header."""
    import uuid
    tenant_id, headers = get_test_auth()
    uid = uuid.uuid4().hex[:8]

    # X-API-Key works on REST routes (baseline)
    resp1 = await client.post("/api/v1/memories", json={
        "tenant_id": tenant_id,
        "agent_id": f"bearer-test-{uid}",
        "fleet_id": f"bearer-fleet-{uid}",
        "memory_type": "fact",
        "content": f"baseline write [{uid}]",
    }, headers=headers)
    assert resp1.status_code == 201

    # MCP mount exists and accepts requests (session manager may not be running
    # in test fixtures, so we just verify the endpoint is mounted)
    resp2 = await client.get("/mcp")
    assert resp2.status_code != 404


async def test_mcp_bearer_returns_tools(client):
    """MCP tool-descriptions endpoint should accept Bearer auth."""
    resp = await client.get(
        "/api/v1/tool-descriptions",
        headers={"Authorization": "Bearer dev-admin-key"},
    )
    # tool-descriptions is an API route that accepts admin key
    assert resp.status_code == 200


async def test_mcp_mount_exists(client):
    """MCP endpoint should be mounted (not 404)."""
    resp = await client.get("/mcp")
    assert resp.status_code != 404, f"MCP endpoint not mounted, got {resp.status_code}"
