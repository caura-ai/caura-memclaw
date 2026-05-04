"""E2E tests for the skill-sharing API.

Exercises POST /api/v1/skills/share + GET /api/v1/skills against a real
docker-compose dev stack via the test client. Verifies:

- The shared skill lands in the ``documents`` collection ``skills``.
- One ``install_skill`` fleet command is enqueued per node in the target fleet.
- Skills are listable and filterable by fleet_id and substring query.
- Validation errors surface as 422 with INVALID_ARGUMENTS.
"""

import pytest

from tests.conftest import get_test_auth, uid as _uid


async def _heartbeat(client, tenant_id, headers, node_name, fleet_id):
    resp = await client.post(
        "/api/v1/fleet/heartbeat",
        json={
            "tenant_id": tenant_id,
            "fleet_id": fleet_id,
            "node_name": node_name,
            "openclaw_version": "1.0.0",
            "os_info": "linux",
            "agents": [],
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _share(client, tenant_id, headers, **overrides):
    body = {
        "tenant_id": tenant_id,
        "name": overrides.pop("name"),
        "description": overrides.pop("description", "A test skill"),
        "content": overrides.pop("content", "# Skill\n\nBody.\n"),
        "target_fleet_id": overrides.pop("target_fleet_id"),
    }
    body.update(overrides)
    return await client.post("/api/v1/skills/share", json=body, headers=headers)


async def test_share_default_publish_only(client):
    """Default share: doc is upserted, NO install_skill commands enqueued."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"
    name = f"probe-{tag}"

    # Register two nodes — they should NOT receive an install command in
    # publish-only mode (the default).
    await _heartbeat(client, tenant_id, headers, f"node-a-{tag}", fleet)
    await _heartbeat(client, tenant_id, headers, f"node-b-{tag}", fleet)

    resp = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        description="Probe skill",
        content="# Probe\nHello.\n",
        target_fleet_id=fleet,
        author_agent_id=f"agent-author-{tag}",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == name
    assert body["install_on_fleet"] is False
    assert body["queued_nodes"] == 0
    assert body["node_ids"] == []

    # Doc still lands in the catalog so teammates can discover it.
    doc_resp = await client.get(
        f"/api/v1/documents/{name}?tenant_id={tenant_id}&collection=skills",
        headers=headers,
    )
    assert doc_resp.status_code == 200, doc_resp.text
    doc = doc_resp.json()
    assert doc["data"]["content"] == "# Probe\nHello.\n"

    # No install_skill commands should exist for this skill.
    cmd_resp = await client.get(
        f"/api/v1/fleet/commands?tenant_id={tenant_id}",
        headers=headers,
    )
    install_cmds = [
        c
        for c in cmd_resp.json()
        if c["command"] == "install_skill" and c["payload"].get("name") == name
    ]
    assert install_cmds == []


async def test_share_install_on_fleet_enqueues_per_node(client):
    """install_on_fleet=true queues an install_skill command per fleet node."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"
    name = f"runbook-{tag}"

    hb_a = await _heartbeat(client, tenant_id, headers, f"node-a-{tag}", fleet)
    hb_b = await _heartbeat(client, tenant_id, headers, f"node-b-{tag}", fleet)
    node_ids = {hb_a["node_id"], hb_b["node_id"]}

    resp = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        target_fleet_id=fleet,
        install_on_fleet=True,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["install_on_fleet"] is True
    assert body["queued_nodes"] == 2
    assert set(body["node_ids"]) == node_ids

    cmd_resp = await client.get(
        f"/api/v1/fleet/commands?tenant_id={tenant_id}",
        headers=headers,
    )
    install_cmds = [
        c
        for c in cmd_resp.json()
        if c["command"] == "install_skill" and c["payload"].get("name") == name
    ]
    assert len(install_cmds) == 2
    assert {c["node_id"] for c in install_cmds} == node_ids
    for c in install_cmds:
        assert c["payload"]["skill_doc_id"] == body["skill_id"]


async def test_share_overwrites_on_resharing(client):
    """Re-sharing the same name upserts (one doc) but enqueues fresh commands."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"
    name = f"probe-{tag}"

    await _heartbeat(client, tenant_id, headers, f"node-{tag}", fleet)

    r1 = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        target_fleet_id=fleet,
        content="v1\n",
        version=1,
    )
    r2 = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        target_fleet_id=fleet,
        content="v2\n",
        version=2,
    )
    assert r1.status_code == 200 and r2.status_code == 200
    # Same skill_id (same doc, upserted) — re-share is idempotent on name.
    assert r1.json()["skill_id"] == r2.json()["skill_id"]

    doc_resp = await client.get(
        f"/api/v1/documents/{name}?tenant_id={tenant_id}&collection=skills",
        headers=headers,
    )
    assert doc_resp.json()["data"]["content"] == "v2\n"


async def test_share_install_on_fleet_with_no_nodes(client):
    """install_on_fleet=true with empty fleet still upserts the doc; queued_nodes=0."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"empty-fleet-{tag}"
    name = f"orphan-{tag}"

    resp = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        target_fleet_id=fleet,
        install_on_fleet=True,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["queued_nodes"] == 0
    assert body["node_ids"] == []


@pytest.mark.parametrize("name", ["", "Has Spaces", "UPPER", "../escape", "x" * 101])
async def test_share_rejects_unsafe_names(client, name):
    """Names that aren't filesystem-safe slugs are rejected as 422."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"

    resp = await _share(
        client,
        tenant_id,
        headers,
        name=name,
        target_fleet_id=fleet,
    )
    # Pydantic catches min_length/max_length first; service catches charset.
    assert resp.status_code == 422, resp.text


async def test_share_requires_target_fleet(client):
    """target_fleet_id is required."""
    tenant_id, headers = get_test_auth()
    tag = _uid()

    resp = await client.post(
        "/api/v1/skills/share",
        json={
            "tenant_id": tenant_id,
            "name": f"probe-{tag}",
            "description": "x",
            "content": "x",
        },
        headers=headers,
    )
    assert resp.status_code == 422


async def test_list_skills_filters_by_fleet(client):
    """GET /skills?fleet_id=X only returns skills shared into X."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet_a = f"fleet-a-{tag}"
    fleet_b = f"fleet-b-{tag}"

    await _share(
        client,
        tenant_id,
        headers,
        name=f"a1-{tag}",
        target_fleet_id=fleet_a,
        description="alpha one",
    )
    await _share(
        client,
        tenant_id,
        headers,
        name=f"a2-{tag}",
        target_fleet_id=fleet_a,
        description="alpha two",
    )
    await _share(
        client,
        tenant_id,
        headers,
        name=f"b1-{tag}",
        target_fleet_id=fleet_b,
        description="beta one",
    )

    resp = await client.get(
        f"/api/v1/skills?tenant_id={tenant_id}&fleet_id={fleet_a}",
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    listing = resp.json()
    names = {s["name"] for s in listing}
    assert f"a1-{tag}" in names
    assert f"a2-{tag}" in names
    assert f"b1-{tag}" not in names


async def test_list_skills_query_returns_matching_skill(client):
    """GET /skills?query=... returns the matching skill.

    With a real embedding provider this is semantic ranked-by-similarity;
    with the fake CI provider it's deterministic noise. We only assert the
    *positive* contract — the relevant skill is discoverable when queried
    by a word in its description — because semantic search ranks rather
    than filters, and unrelated skills can still appear at lower ranks
    (especially with fake embeddings).

    The response carries a ``similarity`` field per row that callers can
    threshold on if they need stricter filtering.
    """
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"

    await _share(
        client,
        tenant_id,
        headers,
        name=f"skill-{tag}",
        target_fleet_id=fleet,
        description="ship-readiness audit recipe",
    )
    await _share(
        client,
        tenant_id,
        headers,
        name=f"recipe-{tag}",
        target_fleet_id=fleet,
        description="unrelated thing",
    )

    resp = await client.get(
        f"/api/v1/skills?tenant_id={tenant_id}&fleet_id={fleet}&query=ship",
        headers=headers,
    )
    assert resp.status_code == 200
    listing = resp.json()
    names = {s["name"] for s in listing}
    assert f"skill-{tag}" in names, "matching skill must surface in query results"


# ---------------------------------------------------------------------------
# DELETE /skills/{name} — unshare
# ---------------------------------------------------------------------------


async def test_unshare_catalog_only(client):
    """DELETE /skills/{name} removes from catalog; no fleet commands by default."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"
    name = f"unshare-cat-{tag}"

    await _heartbeat(client, tenant_id, headers, f"node-{tag}", fleet)
    await _share(client, tenant_id, headers, name=name, target_fleet_id=fleet)

    resp = await client.delete(
        f"/api/v1/skills/{name}?tenant_id={tenant_id}",
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["unshare_from_fleet"] is False
    assert body["queued_nodes"] == 0

    doc_resp = await client.get(
        f"/api/v1/documents/{name}?tenant_id={tenant_id}&collection=skills",
        headers=headers,
    )
    assert doc_resp.status_code == 404

    cmd_resp = await client.get(
        f"/api/v1/fleet/commands?tenant_id={tenant_id}",
        headers=headers,
    )
    uninstalls = [
        c
        for c in cmd_resp.json()
        if c["command"] == "uninstall_skill" and c["payload"].get("name") == name
    ]
    assert uninstalls == []


async def test_unshare_from_fleet_enqueues_uninstall(client):
    """unshare_from_fleet=true queues uninstall_skill per fleet node."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    fleet = f"fleet-{tag}"
    name = f"unshare-fleet-{tag}"

    hb_a = await _heartbeat(client, tenant_id, headers, f"node-a-{tag}", fleet)
    hb_b = await _heartbeat(client, tenant_id, headers, f"node-b-{tag}", fleet)
    expected_nodes = {hb_a["node_id"], hb_b["node_id"]}

    await _share(client, tenant_id, headers, name=name, target_fleet_id=fleet)

    resp = await client.delete(
        f"/api/v1/skills/{name}?tenant_id={tenant_id}"
        f"&target_fleet_id={fleet}&unshare_from_fleet=true",
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["unshare_from_fleet"] is True
    assert body["queued_nodes"] == 2
    assert set(body["node_ids"]) == expected_nodes

    cmd_resp = await client.get(
        f"/api/v1/fleet/commands?tenant_id={tenant_id}",
        headers=headers,
    )
    uninstalls = [
        c
        for c in cmd_resp.json()
        if c["command"] == "uninstall_skill" and c["payload"].get("name") == name
    ]
    assert len(uninstalls) == 2
    assert {c["node_id"] for c in uninstalls} == expected_nodes


async def test_unshare_from_fleet_requires_target_fleet_id(client):
    """unshare_from_fleet=true without target_fleet_id → 422."""
    tenant_id, headers = get_test_auth()
    tag = _uid()
    name = f"missing-fleet-{tag}"

    resp = await client.delete(
        f"/api/v1/skills/{name}?tenant_id={tenant_id}&unshare_from_fleet=true",
        headers=headers,
    )
    assert resp.status_code == 422


async def test_unshare_unknown_skill_returns_deleted_false(client):
    """Unsharing a skill that doesn't exist returns deleted=False (idempotent)."""
    tenant_id, headers = get_test_auth()
    tag = _uid()

    resp = await client.delete(
        f"/api/v1/skills/nonexistent-{tag}?tenant_id={tenant_id}",
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is False
    assert body["queued_nodes"] == 0
