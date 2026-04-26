"""Tests for CAURA-599 — bulk-write atomicity + per-item error contract.

Covers:
- Short-content surfaces as per-item error (not whole-batch 422)
- ``errors: int`` actually counts failures (was dead ``= 0``)
- Reconcile on ``httpx.ReadTimeout`` via content-hash re-query
"""

import uuid

import httpx
import pytest

from tests.conftest import get_test_auth, uid

pytestmark = pytest.mark.asyncio


# ── Per-item short-content errors ──


async def test_short_content_surfaces_per_item_not_whole_batch(client):
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
    resp = await client.post("/api/v1/memories/bulk", json=body, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["created"] == 2
    assert data["errors"] == 1
    assert data["duplicates"] == 0

    by_index = {r["index"]: r for r in data["results"]}
    assert by_index[0]["status"] == "created"
    assert by_index[1]["status"] == "error"
    assert "too short" in by_index[1]["error"]
    assert by_index[2]["status"] == "created"


async def test_all_short_content_batch_returns_all_errors(client):
    tenant_id, headers = get_test_auth()
    body = {
        "tenant_id": tenant_id,
        "agent_id": f"all-short-{uid()}",
        "items": [{"content": "hi"}, {"content": "ok"}, {"content": "yo"}],
    }
    resp = await client.post("/api/v1/memories/bulk", json=body, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["created"] == 0
    assert data["errors"] == 3
    assert data["duplicates"] == 0
    assert all(r["status"] == "error" for r in data["results"])


# ── Reconcile-on-timeout ──


async def test_reconcile_on_timeout_recovers_all_persisted_ids(client, monkeypatch):
    """Storage committed the rows, client timed out reading the response.
    Re-query content hashes → IDs recovered → status='created'."""
    from core_api.clients.storage_client import get_storage_client
    from core_api.services.memory_service import _content_hash

    sc = get_storage_client()
    tenant_id, headers = get_test_auth()
    contents = [f"rc-a-{uid()}", f"rc-b-{uid()}"]
    expected_hashes = [_content_hash(tenant_id, None, c) for c in contents]
    fake_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    async def fake_create(data):
        raise httpx.ReadTimeout("simulated storage timeout")

    calls: list = []

    async def fake_bulk_find(tid, hashes):
        calls.append(list(hashes))
        if len(calls) == 1:
            return {}  # pre-write dedup: nothing exists
        return dict(zip(expected_hashes, fake_ids))  # post-timeout: all persisted

    monkeypatch.setattr(sc, "create_memories", fake_create)
    monkeypatch.setattr(sc, "bulk_find_by_content_hashes", fake_bulk_find)

    body = {
        "tenant_id": tenant_id,
        "agent_id": f"reconcile-{uid()}",
        "items": [{"content": c} for c in contents],
    }
    resp = await client.post("/api/v1/memories/bulk", json=body, headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["created"] == 2
    assert data["errors"] == 0
    assert len(calls) == 2  # pre-write + post-timeout


async def test_reconcile_marks_unpersisted_items_as_error(client, monkeypatch):
    """Partial-timeout reconcile: only some hashes came back from storage.
    Recovered items → 'created'; unrecovered → 'error: timeout unknown'."""
    from core_api.clients.storage_client import get_storage_client
    from core_api.services.memory_service import _content_hash

    sc = get_storage_client()
    tenant_id, headers = get_test_auth()
    contents = [f"rcp-a-{uid()}", f"rcp-b-{uid()}"]
    expected_hashes = [_content_hash(tenant_id, None, c) for c in contents]

    async def fake_create(data):
        raise httpx.ReadTimeout("simulated partial")

    calls: list = []

    async def fake_bulk_find(tid, hashes):
        calls.append(list(hashes))
        if len(calls) == 1:
            return {}
        # Only item 0 landed in storage before the timeout.
        return {expected_hashes[0]: str(uuid.uuid4())}

    monkeypatch.setattr(sc, "create_memories", fake_create)
    monkeypatch.setattr(sc, "bulk_find_by_content_hashes", fake_bulk_find)

    body = {
        "tenant_id": tenant_id,
        "agent_id": f"reconcile-partial-{uid()}",
        "items": [{"content": c} for c in contents],
    }
    resp = await client.post("/api/v1/memories/bulk", json=body, headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["created"] == 1
    assert data["errors"] == 1

    by_index = {r["index"]: r for r in data["results"]}
    assert by_index[0]["status"] == "created"
    assert by_index[1]["status"] == "error"
    assert "timeout" in by_index[1]["error"].lower()
