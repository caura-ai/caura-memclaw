"""Tests for CAURA-603 part B — slowapi rate limiter.

Exercises the limiter at the HTTP layer (decorators applied to
/memories, /memories/bulk, /search). The in-memory slowapi backend is
used here so tests don't need Redis; prod swaps to Redis via
``settings.redis_url``.

Tests tune ``rate_limit_*`` settings to tiny values so a 429 fires
within a handful of rapid requests. Each test also resets the limiter's
in-memory store afterwards so neighbours stay independent.
"""

import pytest

from core_api.middleware.rate_limit import _key_func, limiter

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _enable_limiter_for_rate_tests():
    """Overrides the session-scoped conftest fixture that disables the
    limiter for the rest of the suite. Clears buckets between tests so
    earlier bursts don't leak state into later tests."""
    prev = limiter.enabled
    limiter.enabled = True
    yield
    limiter.enabled = prev
    limiter.reset()


# ── key_func ──


class _FakeRequest:
    def __init__(self, headers: dict[str, str], client_host: str = "127.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": client_host})()
        # slowapi's get_remote_address looks at scope too; emulate it.
        self.scope = {"client": (client_host, 0)}


async def test_key_func_prefers_api_key_over_ip():
    # API-key path produces a `key:<32-hex>` token, not the raw key.
    req = _FakeRequest(headers={"x-api-key": "mc_abcdef12345678_secret"})
    out = _key_func(req)
    assert out.startswith("key:")
    assert len(out) == len("key:") + 32
    # Secret must NOT appear verbatim — it's hashed.
    assert "mc_abcdef" not in out


async def test_key_func_accepts_bearer_auth():
    req = _FakeRequest(headers={"authorization": "Bearer mc_bearertoken_abcde"})
    out = _key_func(req)
    assert out.startswith("key:")
    assert "bearertoken" not in out


async def test_key_func_different_keys_yield_different_buckets():
    # Keys sharing a 16-char prefix used to collide; hashing fixes that.
    a = _FakeRequest(headers={"x-api-key": "mc_samentenantid_A"})
    b = _FakeRequest(headers={"x-api-key": "mc_samentenantid_B"})
    assert _key_func(a) != _key_func(b)


async def test_key_func_falls_back_to_ip_without_api_key():
    req = _FakeRequest(headers={}, client_host="198.51.100.7")
    assert _key_func(req) == "ip:198.51.100.7"


# ── HTTP layer: 429 on burst ──


async def test_search_rate_limit_returns_429_after_budget(client):
    """The default 30/second limit should trip within a 100-request burst
    from a single API key."""
    headers = {"x-api-key": "mc_rate_test_search_key"}
    body = {"tenant_id": "default", "query": "hello"}
    codes = []
    for _ in range(100):
        resp = await client.post("/api/v1/search", json=body, headers=headers)
        codes.append(resp.status_code)
    assert 429 in codes, f"expected at least one 429 among {codes}"


async def test_write_rate_limit_returns_429_after_budget(client):
    """Writes have a stricter limit (10/second by default) — a 50-request
    burst from the same key should reliably see a 429."""
    headers = {"x-api-key": "mc_rate_test_write_key"}
    body = {
        "tenant_id": "default",
        "agent_id": "rate-test-agent",
        "memory_type": "fact",
        "content": "rate-limit probe",
    }
    codes = []
    for i in range(50):
        body["content"] = f"rate-limit probe {i}"  # avoid dedup 409s
        resp = await client.post("/api/v1/memories", json=body, headers=headers)
        codes.append(resp.status_code)
    assert 429 in codes, f"expected at least one 429 among {codes}"


async def test_health_is_not_rate_limited(client):
    """/health runs pre-auth and must not be rate-limited or the deploy
    gate (CAURA-603 part A) stops working under load."""
    codes = []
    for _ in range(60):
        resp = await client.get("/api/v1/health")
        codes.append(resp.status_code)
    assert 429 not in codes
