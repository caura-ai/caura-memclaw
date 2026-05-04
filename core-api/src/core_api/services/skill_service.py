"""Skill sharing — agent-to-agent SKILL.md distribution across the fleet.

Shared by REST (`POST /api/v1/skills/share`, `GET /api/v1/skills`,
`DELETE /api/v1/skills/{name}`) and MCP (`memclaw_share_skill`,
`memclaw_unshare_skill`). Stores skills as documents in a fixed
``skills`` collection (with the ``description`` field embedded for
semantic search) and distributes them via the existing
``fleet_commands`` queue.

Design notes:

- Skills are NOT memories. They have no weight decay, no supersession
  chain, no recall_count — they're stable reference docs that an agent
  reads on demand. Storing them in ``documents`` (collection=``skills``)
  reuses the structured-record machinery and keeps the recall layer
  free of skill bodies.
- Description is auto-embedded so ``GET /skills?query=`` and
  ``memclaw_doc op=search collection=skills`` retrieve skills by
  meaning, not just by exact name match.
- Distribution piggybacks on ``fleet_commands``: ``install_skill`` /
  ``uninstall_skill`` payloads ride the same queue as today's
  ``educate``/``deploy``/``ping``/``restart``.
- ``target_fleet_id`` is the routing key (single fleet per share).
  ``target_agent_ids`` is informational only in v1.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core_api.clients.storage_client import get_storage_client
from core_api.repositories import document_repo

logger = logging.getLogger(__name__)


SKILLS_COLLECTION = "skills"
"""Fixed collection name for shared skills. Centralised so REST + MCP
agree on where to read/write."""

INSTALL_SKILL_COMMAND = "install_skill"
UNINSTALL_SKILL_COMMAND = "uninstall_skill"
"""Fleet-command names the plugin's heartbeat handler dispatches on
(`plugin/src/heartbeat.ts:processCommand`)."""

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
"""Skill names double as directory names on the plugin side — keep them
filesystem-safe and short. Lowercase, alphanumeric + ``.``/``_``/``-``,
1-100 chars, must start with alphanumeric."""


def validate_skill_name(name: str) -> None:
    """Raise ``ValueError`` if ``name`` isn't a safe skill identifier."""
    if not _NAME_RE.fullmatch(name or ""):
        raise ValueError(
            f"Invalid skill name {name!r}. Must match {_NAME_RE.pattern} — "
            "lowercase alphanumerics plus '.', '_', '-' (max 100 chars, "
            "must start with alphanumeric)."
        )


async def share_skill(
    *,
    db: AsyncSession,
    tenant_id: str,
    name: str,
    description: str,
    content: str,
    target_fleet_id: str,
    install_on_fleet: bool = False,
    author_agent_id: str | None = None,
    target_agent_ids: list[str] | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """Upsert a skill into the shared collection; optionally queue install commands.

    The ``description`` field is embedded at write time so the skill is
    discoverable via semantic search (``GET /skills?query=...`` and
    ``memclaw_doc op=search collection=skills``).

    Two distribution modes:

    - ``install_on_fleet=False`` (default) — **publish-only**. Skill
      lands in the catalog; recipients pull on demand via
      ``memclaw_doc op=query collection=skills`` or semantic search.
    - ``install_on_fleet=True`` — **push to every node** in
      ``target_fleet_id`` via ``install_skill`` fleet commands.

    Returns ``{"skill_id", "doc_id", "name", "target_fleet_id",
    "install_on_fleet", "queued_nodes", "node_ids"}``.

    Raises ``ValueError`` for invalid input.
    """
    validate_skill_name(name)
    if not content or not content.strip():
        raise ValueError("Skill content is required and must be non-empty.")
    if not description or not description.strip():
        raise ValueError("Skill description is required.")
    if not target_fleet_id:
        raise ValueError("target_fleet_id is required.")

    data = {
        "name": name,
        "description": description,
        "content": content,
        "author_agent_id": author_agent_id,
        "target_fleet_id": target_fleet_id,
        "target_agent_ids": list(target_agent_ids or []),
        "version": version,
    }

    # Embed the description for semantic discovery. Failures are logged
    # but don't fail the share — substring-matching the catalog still
    # works. ``get_embedding`` returns ``None`` when the provider is
    # unconfigured (e.g. fake provider in tests); treat that as "skip
    # indexing" rather than an error.
    embedding: list[float] | None = None
    try:
        from common.embedding import get_embedding

        embedding = await get_embedding(description)
    except Exception:
        logger.warning(
            "share_skill: failed to embed description for skill %s; "
            "skill is still browseable via name/substring filter",
            name,
            exc_info=True,
        )

    row = await document_repo.upsert_returning_xmax(
        db,
        tenant_id=tenant_id,
        fleet_id=target_fleet_id,
        collection=SKILLS_COLLECTION,
        doc_id=name,
        data=data,
        embedding=embedding,
    )
    if row is None:
        raise RuntimeError("skill upsert returned no rows")
    skill_id = str(row[0])
    await db.commit()

    # Publish-only mode short-circuits before touching fleet_commands.
    node_ids: list[str] = []
    if install_on_fleet:
        sc = get_storage_client()
        nodes = await sc.list_nodes(tenant_id=tenant_id, fleet_id=target_fleet_id)
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            try:
                await sc.create_command(
                    {
                        "tenant_id": tenant_id,
                        "node_id": str(node_id),
                        "command": INSTALL_SKILL_COMMAND,
                        "payload": {
                            "skill_doc_id": skill_id,
                            "name": name,
                            "version": version,
                        },
                    }
                )
                node_ids.append(str(node_id))
            except Exception:
                logger.warning(
                    "share_skill: failed to enqueue install_skill on node %s for skill %s",
                    node_id,
                    name,
                    exc_info=True,
                )

    return {
        "skill_id": skill_id,
        "doc_id": name,
        "name": name,
        "target_fleet_id": target_fleet_id,
        "install_on_fleet": install_on_fleet,
        "queued_nodes": len(node_ids),
        "node_ids": node_ids,
    }


async def unshare_skill(
    *,
    db: AsyncSession,
    tenant_id: str,
    name: str,
    unshare_from_fleet: bool = False,
    target_fleet_id: str | None = None,
) -> dict[str, Any]:
    """Remove a skill from the catalog; optionally uninstall from fleet nodes.

    Two modes (mirror of ``share_skill``):

    - ``unshare_from_fleet=False`` (default) — **catalog-only removal**.
      Doc is deleted from the ``skills`` collection so new agents won't
      discover it; nodes that already installed the skill keep their
      local copy until they're cleaned up out-of-band.
    - ``unshare_from_fleet=True`` — also queue ``uninstall_skill`` fleet
      commands per node in ``target_fleet_id`` (required in this mode).
      Plugin handlers ``rm`` the local SKILL.md (idempotent — succeeds
      whether or not the file exists).

    Returns ``{"name", "deleted": bool, "unshare_from_fleet", "queued_nodes",
    "node_ids"}``. ``deleted=False`` means the skill wasn't in the catalog
    (nothing to do) — still success since the end state is "skill not
    present".
    """
    validate_skill_name(name)
    if unshare_from_fleet and not target_fleet_id:
        raise ValueError("target_fleet_id is required when unshare_from_fleet=true.")

    sc = get_storage_client()
    deleted = await sc.delete_document(tenant_id=tenant_id, collection=SKILLS_COLLECTION, doc_id=name)

    node_ids: list[str] = []
    if unshare_from_fleet and target_fleet_id:
        nodes = await sc.list_nodes(tenant_id=tenant_id, fleet_id=target_fleet_id)
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                continue
            try:
                await sc.create_command(
                    {
                        "tenant_id": tenant_id,
                        "node_id": str(node_id),
                        "command": UNINSTALL_SKILL_COMMAND,
                        "payload": {"name": name},
                    }
                )
                node_ids.append(str(node_id))
            except Exception:
                logger.warning(
                    "unshare_skill: failed to enqueue uninstall_skill on node %s for skill %s",
                    node_id,
                    name,
                    exc_info=True,
                )

    return {
        "name": name,
        "deleted": bool(deleted),
        "unshare_from_fleet": unshare_from_fleet,
        "target_fleet_id": target_fleet_id,
        "queued_nodes": len(node_ids),
        "node_ids": node_ids,
    }


async def list_skills(
    *,
    db: AsyncSession,
    tenant_id: str,
    fleet_id: str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List skills available to the caller.

    When ``query`` is supplied, runs a semantic search over the
    embedded ``description`` field and returns top-``limit`` matches
    (``offset`` ignored — semantic search is ranked, not paginated).
    When ``query`` is omitted, returns the most-recently-updated skills,
    optionally filtered by ``fleet_id``.

    Each summary carries the embedding similarity (when applicable) so
    clients can show ranked hits — the ``content`` field is intentionally
    omitted; fetch the full body via ``GET /documents/<id>?collection=skills``
    if needed.
    """
    q = (query or "").strip()
    if q:
        # Semantic path — embed the query and use the partial HNSW index.
        from common.embedding import get_embedding

        q_embedding = await get_embedding(q)
        if q_embedding is None:
            # Embedding unavailable (no provider configured); fall back
            # to substring filter so callers without an embedding
            # provider still get useful results.
            return await _list_skills_substring(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                substring=q,
                limit=limit,
                offset=offset,
            )
        rows = await document_repo.search(
            db,
            tenant_id=tenant_id,
            query_embedding=q_embedding,
            collection=SKILLS_COLLECTION,
            top_k=max(limit, 1),
            fleet_id=fleet_id,
        )
        return [_doc_to_summary(doc, similarity=sim) for doc, sim in rows]

    # No query — list by recency.
    sc = get_storage_client()
    docs = await sc.query_documents(
        {
            "tenant_id": tenant_id,
            "collection": SKILLS_COLLECTION,
            "fleet_id": fleet_id,
            "where": {},
            "limit": limit,
            "offset": offset,
        }
    )
    return [_doc_to_summary_dict(d) for d in docs]


async def _list_skills_substring(
    *,
    tenant_id: str,
    fleet_id: str | None,
    substring: str,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    """Substring-filter fallback when no embedding provider is available."""
    sc = get_storage_client()
    docs = await sc.query_documents(
        {
            "tenant_id": tenant_id,
            "collection": SKILLS_COLLECTION,
            "fleet_id": fleet_id,
            "where": {},
            "limit": limit,
            "offset": offset,
        }
    )
    s = substring.lower()
    out: list[dict[str, Any]] = []
    for d in docs:
        data = d.get("data") or {}
        name = str(data.get("name") or d.get("doc_id") or "")
        description = str(data.get("description") or "")
        if s in name.lower() or s in description.lower():
            out.append(_doc_to_summary_dict(d))
    return out


def _doc_to_summary(doc, similarity: float | None = None) -> dict[str, Any]:
    """Format a Document ORM row as a summary dict (no content body)."""
    data = doc.data or {}
    out: dict[str, Any] = {
        "skill_id": str(doc.id),
        "name": str(data.get("name") or doc.doc_id or ""),
        "description": str(data.get("description") or ""),
        "author_agent_id": data.get("author_agent_id"),
        "target_fleet_id": data.get("target_fleet_id"),
        "target_agent_ids": data.get("target_agent_ids") or [],
        "version": int(data.get("version") or 1),
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }
    if similarity is not None:
        out["similarity"] = round(float(similarity), 4)
    return out


def _doc_to_summary_dict(d: dict) -> dict[str, Any]:
    """Format a storage_client doc dict (returned by ``query_documents``)
    as a summary. Used by the non-semantic listing path."""
    data = d.get("data") or {}
    return {
        "skill_id": str(d.get("id", "")),
        "name": str(data.get("name") or d.get("doc_id") or ""),
        "description": str(data.get("description") or ""),
        "author_agent_id": data.get("author_agent_id"),
        "target_fleet_id": data.get("target_fleet_id"),
        "target_agent_ids": data.get("target_agent_ids") or [],
        "version": int(data.get("version") or 1),
        "created_at": str(d["created_at"]) if d.get("created_at") else None,
        "updated_at": str(d["updated_at"]) if d.get("updated_at") else None,
    }
