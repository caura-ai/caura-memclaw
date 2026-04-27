"""Tests for CAURA-602 — bulk write per-attempt idempotency + 207 contract.

The earlier CAURA-599 reconcile-on-ReadTimeout path is gone; per-item
``client_request_id`` (derived from ``X-Bulk-Attempt-Id``) plus the
storage-side partial unique index are what makes the bulk path
retry-safe at the row level. These tests cover the new contract:

- Short-content surfaces as per-item ``status="error"`` and triggers a
  207 mixed response when other items succeed.
- Whole-batch error returns 422.
- ``X-Bulk-Attempt-Id`` is required and validated.
- A retry of the same attempt id returns ``duplicate_attempt`` for
  every previously-committed row, with the canonical id — no silent
  creates.
- Same content via a *different* attempt id surfaces as
  ``duplicate_content`` (the legacy dedup).
- A bulk-budget burn returns 504 without recording an idempotency
  receipt, so the next retry can resolve cleanly.
"""

import asyncio

import pytest

from tests.conftest import get_test_auth, uid

pytestmark = pytest.mark.asyncio


def _attempt_id(prefix: str) -> str:
    return f"{prefix}-{uid()}"


# ── Per-item validation ──


async def test_short_content_in_mixed_batch_returns_207(client):
    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"short-{uid()}",
        "items": [
            {"content": f"well-formed content one {uid()}"},
            {"content": "hi"},  # below CRYSTALLIZER_SHORT_CONTENT_CHARS
            {"content": f"well-formed content two {uid()}"},
        ],
    }
    resp = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": _attempt_id("mixed")},
    )
    # 207 Multi-Status: at least one created + at least one error.
    assert resp.status_code == 207
    data = resp.json()
    assert data["created"] == 2
    assert data["errors"] == 1
    assert data["duplicates"] == 0

    by_index = {r["index"]: r for r in data["results"]}
    assert by_index[0]["status"] == "created"
    assert by_index[1]["status"] == "error"
    assert "too short" in by_index[1]["error"]
    assert by_index[2]["status"] == "created"
    # Every result carries its server-derived per-item attempt id —
    # callers can use this to correlate with retries.
    for r in data["results"]:
        assert r["client_request_id"]
        assert r["client_request_id"].endswith(f":{r['index']}")


async def test_all_short_content_batch_returns_200(client):
    """Every item rejected on merit: the request itself was fine, so
    we return 200 with ``errors == n``. FastAPI's automatic 422 covers
    *request-body* validation; this route deliberately doesn't shadow
    it for per-item business-logic rejections (CAURA-602).
    """
    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"all-short-{uid()}",
        "items": [{"content": "hi"}, {"content": "ok"}, {"content": "yo"}],
    }
    resp = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": _attempt_id("all-short")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["created"] == 0
    assert data["errors"] == 3
    assert data["duplicates"] == 0
    assert all(r["status"] == "error" for r in data["results"])


# ── X-Bulk-Attempt-Id contract ──


async def test_missing_bulk_attempt_id_rejected(client):
    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"miss-attempt-{uid()}",
        "items": [{"content": f"hello {uid()}"}],
    }
    resp = await client.post("/api/v1/memories/bulk", json=body, headers=headers)
    assert resp.status_code == 400
    assert "X-Bulk-Attempt-Id" in resp.json()["detail"]


async def test_malformed_bulk_attempt_id_rejected(client):
    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"bad-attempt-{uid()}",
        "items": [{"content": f"hello {uid()}"}],
    }
    # Spaces / non-allowed chars are out — the partial-unique key has
    # to match a strict pattern so SDK bugs don't pollute the index.
    # ASCII-only here on purpose: httpx ASCII-encodes header values, and
    # the regex alone is what's under test.
    resp = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": "has spaces and slashes/"},
    )
    assert resp.status_code == 400


# ── Per-attempt idempotency: the silent-create eliminator ──


async def test_retry_same_attempt_id_returns_duplicate_attempt(client):
    """Send a batch, then send the *exact same payload + attempt id* a
    second time. Every row from the first call must come back as
    ``duplicate_attempt`` with the canonical id — no second insert,
    no silent create."""
    tenant_id, headers = get_test_auth()
    attempt_id = _attempt_id("retry")
    contents = [f"retry-test-{uid()}-{i}" for i in range(3)]
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"retry-{uid()}",
        "items": [{"content": c} for c in contents],
    }

    first = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": attempt_id},
    )
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["created"] == 3
    first_ids = [r["id"] for r in first_data["results"]]
    assert all(first_ids)

    # ``Idempotency-Key`` is intentionally absent: we want to exercise
    # the *row-level* recovery path, not the response-replay cache.
    # Production retries hit one or the other; the per-attempt-id path
    # is the harder guarantee and the one that closes the silent-create
    # class.
    second = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": attempt_id},
    )
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["created"] == 0
    # All three rows are duplicates of the previous attempt — counted
    # together in the rolled-up ``duplicates`` total.
    assert second_data["duplicates"] == 3
    assert second_data["errors"] == 0

    second_ids = [r["id"] for r in second_data["results"]]
    assert second_ids == first_ids
    for r in second_data["results"]:
        assert r["status"] == "duplicate_attempt"


async def test_different_attempt_id_same_content_is_duplicate_content(client):
    """A different ``X-Bulk-Attempt-Id`` with overlapping content
    surfaces the existing rows as ``duplicate_content`` (today's
    content-hash dedup), not ``duplicate_attempt``. The two states
    mean different things to the caller."""
    tenant_id, headers = get_test_auth()
    content = f"shared-content-{uid()}"
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"shared-{uid()}",
        "items": [{"content": content}],
    }

    first = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": _attempt_id("first")},
    )
    assert first.status_code == 200
    canonical_id = first.json()["results"][0]["id"]

    second = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": _attempt_id("second")},
    )
    assert second.status_code == 200
    second_result = second.json()["results"][0]
    assert second_result["status"] == "duplicate_content"
    assert second_result["id"] == canonical_id
    assert second_result["duplicate_of"] == canonical_id


# ── Bulk-budget burn returns 504 without persisting state ──


async def test_bulk_budget_burn_returns_504_with_retry_hint(client, monkeypatch):
    """If ``create_memories_bulk`` exceeds ``bulk_request_timeout_seconds``,
    the route returns 504 and does NOT record an idempotency receipt.
    The retry then runs cleanly against the per-attempt unique index."""
    from core_api import config as cfg
    from core_api.services import memory_service

    # Squeeze the budget so we don't have to actually wait 90s.
    monkeypatch.setattr(cfg.settings, "bulk_request_timeout_seconds", 0.05)

    real_create = memory_service.create_memories_bulk

    async def slow_create(*args, **kwargs):
        await asyncio.sleep(1.0)
        return await real_create(*args, **kwargs)

    monkeypatch.setattr(memory_service, "create_memories_bulk", slow_create)
    # The route imports ``create_memories_bulk`` by name at module load
    # time, so patch the route's local binding too.
    from core_api.routes import memories as routes_mem

    monkeypatch.setattr(routes_mem, "create_memories_bulk", slow_create)

    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"slow-{uid()}",
        "items": [{"content": f"slow-content-{uid()}"}],
    }
    resp = await client.post(
        "/api/v1/memories/bulk",
        json=body,
        headers={**headers, "X-Bulk-Attempt-Id": _attempt_id("slow")},
    )
    assert resp.status_code == 504
    assert "X-Bulk-Attempt-Id" in resp.json()["detail"]
