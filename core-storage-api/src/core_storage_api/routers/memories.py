"""Memory CRUD, search, lifecycle, and dedup endpoints."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from core_storage_api.observability import bind_timer, log_request
from core_storage_api.schemas import MEMORY_FIELDS, orm_to_dict
from core_storage_api.services.postgres_service import PostgresService

router = APIRouter(prefix="/memories", tags=["Memories"])
_svc = PostgresService()


# ------------------------------------------------------------------
# Core CRUD (non-parameterised paths first)
# ------------------------------------------------------------------


_DATETIME_FIELDS = {
    "created_at",
    "expires_at",
    "deleted_at",
    "last_recalled_at",
    "ts_valid_start",
    "ts_valid_end",
    "last_dedup_checked_at",
}


def _parse_datetimes(body: dict) -> dict:
    """Convert ISO-format datetime strings to ``datetime`` objects.

    Malformed ISO strings raise ``HTTPException(422)`` rather than
    propagating the ``ValueError`` from ``datetime.fromisoformat`` as
    a 500 — a request whose body says ``"ts_valid_start": "tomorrow"``
    is a client validation problem, not a server fault. Applies to
    both POST and PATCH routes since both share this helper.
    """
    for key in _DATETIME_FIELDS:
        val = body.get(key)
        if isinstance(val, str):
            try:
                body[key] = datetime.fromisoformat(val)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid ISO datetime for field {key!r}: {val!r}",
                )
    return body


@router.post("")
async def create_memory(request: Request) -> dict:
    body: dict = await request.json()
    _parse_datetimes(body)
    memory = await _svc.memory_add(body)
    return orm_to_dict(memory, MEMORY_FIELDS)


@router.post("/bulk")
async def create_memories_bulk(request: Request) -> list[dict]:
    """Insert a batch with per-attempt idempotency (CAURA-602).

    Each item must carry ``client_request_id`` (server-derived from
    ``X-Bulk-Attempt-Id`` upstream, or a UUID for in-process callers
    like auto-chunk). The response is per-item — ``{client_request_id,
    id, was_inserted}`` in input order — so the upstream core-api can
    map to ``created`` (was_inserted=True) vs ``duplicate_attempt``
    (False) without a second roundtrip. The full ORM dict was the prior
    contract; downstream callers reconstruct any other fields from the
    request payload they already hold.
    """
    body: list[dict] = await request.json()
    for item in body:
        _parse_datetimes(item)
    return await _svc.memory_add_all(body)


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


@router.post("/scored-search")
async def scored_search(request: Request) -> list[dict]:
    body: dict = await request.json()
    # Build search_params from top-level body keys (client sends them flat)
    _SEARCH_PARAM_KEYS = {
        "fts_weight",
        "freshness_floor",
        "freshness_decay_days",
        "recall_boost_cap",
        "recall_decay_window_days",
        "similarity_blend",
    }
    search_params = body.get("search_params") or {k: body[k] for k in _SEARCH_PARAM_KEYS if k in body}

    # Parse temporal_window from days (legacy) or seconds (pipeline path)
    temporal_window = None
    if body.get("temporal_window_days"):
        temporal_window = timedelta(days=body["temporal_window_days"])
    elif body.get("temporal_window_seconds"):
        temporal_window = timedelta(seconds=body["temporal_window_seconds"])

    # Hard date-range filter (pipeline path)
    date_range_start = body.get("date_range_start")
    date_range_end = body.get("date_range_end")

    # Parse valid_at ISO string to datetime
    valid_at = body.get("valid_at")
    if isinstance(valid_at, str):
        valid_at = datetime.fromisoformat(valid_at)

    t_start = time.perf_counter()
    db_timer = None
    out: list[dict] = []
    success = True
    try:
        with bind_timer() as db_timer:
            results = await _svc.memory_scored_search(
                tenant_id=body["tenant_id"],
                embedding=body["embedding"],
                query=body["query"],
                fleet_ids=body.get("fleet_ids"),
                caller_agent_id=body.get("caller_agent_id"),
                filter_agent_id=body.get("filter_agent_id"),
                memory_type_filter=body.get("memory_type_filter"),
                status_filter=body.get("status_filter"),
                valid_at=valid_at,
                boosted_memory_ids=set(body["boosted_memory_ids"])
                if body.get("boosted_memory_ids")
                else None,
                memory_boost_factor={UUID(k): v for k, v in body["memory_boost_factor"].items()}
                if body.get("memory_boost_factor")
                else None,
                search_params=search_params,
                temporal_window=temporal_window,
                recall_boost_enabled=body.get("recall_boost_enabled", True),
                top_k=body.get("top_k", 10),
                date_range_start=date_range_start,
                date_range_end=date_range_end,
            )
        for r in results:
            row = orm_to_dict(r.Memory, MEMORY_FIELDS)
            row["score"] = float(r.score) if r.score is not None else 0.0
            row["similarity"] = float(r.similarity) if r.similarity is not None else 0.0
            row["vec_sim"] = float(r.vec_sim) if r.vec_sim is not None else 0.0
            # CAURA-594: authoritative signal for async-embed callers;
            # `vec_sim == 0.0` is ambiguous with an orthogonal embedding.
            # Default to False on missing attribute — the only readers
            # are workers deciding whether to re-embed, and a redundant
            # re-embed is harmless while a silent skip of a NULL row is
            # not.
            row["has_embedding"] = bool(getattr(r, "has_embedding", False))
            row["status_penalty"] = (
                float(r.status_penalty) if getattr(r, "status_penalty", None) is not None else 1.0
            )
            row["entity_links"] = r.entity_links or []
            out.append(row)
    except Exception:
        success = False
        raise
    finally:
        # body.get() here — never indexed access — so a malformed payload
        # that omits tenant_id doesn't raise a secondary KeyError in the
        # finally and swallow the original exception.
        log_request(
            "scored-search",
            tenant_id=body.get("tenant_id"),
            top_k=body.get("top_k", 10),
            total_ms=(time.perf_counter() - t_start) * 1000,
            db_ms=db_timer.total_ms if db_timer is not None else 0.0,
            row_count=len(out),
            has_date_range=bool(date_range_start and date_range_end),
            has_temporal_window=temporal_window is not None,
            error=not success,
        )
    return out


# ------------------------------------------------------------------
# Dedup / content hash
# ------------------------------------------------------------------


@router.post("/semantic-duplicate")
async def find_semantic_duplicate(request: Request) -> dict:
    body: dict = await request.json()
    memory = await _svc.memory_find_semantic_duplicate(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        embedding=body["embedding"],
        exclude_id=UUID(body["exclude_id"]) if body.get("exclude_id") else None,
        visibility=body.get("visibility"),
    )
    if memory is None:
        raise HTTPException(status_code=404, detail="No semantic duplicate found")
    return orm_to_dict(memory, MEMORY_FIELDS)


@router.post("/entity-overlap-candidates")
async def find_entity_overlap_candidates(request: Request) -> list[dict]:
    body: dict = await request.json()
    memories = await _svc.memory_find_entity_overlap_candidates(
        memory_id=UUID(body["memory_id"]),
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        limit=body.get("limit", 8),
    )
    return [orm_to_dict(m, MEMORY_FIELDS) for m in memories]


@router.post("/find-successors")
async def find_successors(request: Request) -> list[dict]:
    body: dict = await request.json()
    valid_at = body.get("valid_at")
    if isinstance(valid_at, str):
        from datetime import datetime

        valid_at = datetime.fromisoformat(valid_at)
    memories = await _svc.memory_find_successors(
        supersedes_ids=[UUID(sid) for sid in body["supersedes_ids"]],
        tenant_id=body["tenant_id"],
        fleet_ids=body.get("fleet_ids"),
        caller_agent_id=body.get("caller_agent_id"),
        filter_agent_id=body.get("filter_agent_id"),
        memory_type_filter=body.get("memory_type_filter"),
        valid_at=valid_at,
    )
    return [orm_to_dict(m, MEMORY_FIELDS) for m in memories]


@router.post("/similar-candidates")
async def find_similar_candidates(request: Request) -> list[dict]:
    body: dict = await request.json()
    memories = await _svc.memory_find_similar_candidates(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        embedding=body["embedding"],
        memory_id=UUID(body["memory_id"]),
        visibility=body.get("visibility", "scope_team"),
        threshold=body.get("threshold", 0.7),
        limit=body.get("limit", 20),
    )
    return [orm_to_dict(m, MEMORY_FIELDS) for m in memories]


@router.get("/by-content-hash")
async def find_by_content_hash(
    tenant_id: str,
    content_hash: str,
    fleet_id: str | None = None,
) -> dict:
    memory = await _svc.memory_find_by_content_hash(tenant_id, content_hash, fleet_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found by content hash")
    return orm_to_dict(memory, MEMORY_FIELDS)


@router.get("/embedding-by-content-hash")
async def find_embedding_by_content_hash(
    tenant_id: str,
    content_hash: str,
) -> list[float] | None:
    return await _svc.memory_find_embedding_by_content_hash(tenant_id, content_hash)


@router.get("/duplicate-hash")
async def find_duplicate_hash(
    tenant_id: str,
    content_hash: str,
    exclude_id: str | None = None,
) -> dict | None:
    dup_id = await _svc.memory_find_duplicate_hash(
        tenant_id,
        content_hash,
        exclude_id=UUID(exclude_id) if exclude_id else None,
    )
    if dup_id is None:
        return None
    return {"memory_id": str(dup_id)}


@router.post("/bulk-by-content-hashes")
async def bulk_find_by_content_hashes(request: Request) -> dict:
    """Wire format: ``{content_hash: {id, client_request_id}}``.

    See ``memory_bulk_find_by_content_hashes`` for why
    ``client_request_id`` is part of the response — the upstream bulk
    route uses it to distinguish ``duplicate_attempt`` from
    ``duplicate_content`` (CAURA-602).
    """
    body: dict = await request.json()
    result = await _svc.memory_bulk_find_by_content_hashes(
        tenant_id=body["tenant_id"],
        hashes=body["hashes"],
    )
    return {ch: {"id": str(v["id"]), "client_request_id": v["client_request_id"]} for ch, v in result.items()}


@router.get("/rdf-conflicts")
async def find_rdf_conflicts(
    tenant_id: str,
    subject_entity_id: str,
    predicate: str,
    exclude_id: str | None = None,
) -> list[dict]:
    memories = await _svc.memory_find_rdf_conflicts(
        tenant_id=tenant_id,
        subject_entity_id=UUID(subject_entity_id),
        predicate=predicate,
        # Service requires object_value and memory_id; use empty/exclude_id defaults
        # so the endpoint works as a conflict finder by subject+predicate.
        object_value="",
        memory_id=UUID(exclude_id) if exclude_id else UUID(int=0),
    )
    return [orm_to_dict(m, MEMORY_FIELDS) for m in memories]


@router.post("/near-duplicates")
async def check_near_duplicates(request: Request) -> dict:
    body: dict = await request.json()
    candidates = await _svc.memory_find_near_duplicate_candidates(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        batch_size=body.get("batch_size", 100),
        offset=body.get("offset", 0),
    )
    return {"candidates": [{"id": str(r[0]), "embedding": r[1]} for r in candidates]}


@router.post("/neighbors-by-embedding")
async def find_neighbors_by_embedding(request: Request) -> list[dict]:
    body: dict = await request.json()
    rows = await _svc.memory_find_neighbors_by_embedding(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        query_embedding=body["query_embedding"],
        exclude_id=UUID(body["exclude_id"]),
        threshold=body.get("threshold", 0.95),
        limit=body.get("limit", 5),
    )
    return [{"id": str(r[0]), "similarity": float(r[1])} for r in rows]


@router.post("/mark-dedup-checked")
async def mark_dedup_checked(request: Request) -> dict:
    body: dict = await request.json()
    memory_ids = [UUID(mid) for mid in body["memory_ids"]]
    await _svc.memory_mark_dedup_checked(memory_ids)
    return {"ok": True}


@router.post("/entity-links")
async def get_entity_links_for_memories(request: Request) -> dict:
    body: dict = await request.json()
    memory_ids = [UUID(mid) for mid in body["memory_ids"]]
    links = await _svc.memory_get_entity_links_for_memories(memory_ids)
    # Serialise UUID keys to strings
    return {
        str(k): [{"entity_id": str(el["entity_id"]), "role": el["role"]} for el in v]
        for k, v in links.items()
    }


# ------------------------------------------------------------------
# Batch status
# ------------------------------------------------------------------


@router.post("/batch-update-status")
async def batch_update_status(request: Request) -> dict:
    body: dict = await request.json()
    for item in body.get("updates", []):
        await _svc.memory_update_status(UUID(item["memory_id"]), item["status"])
    return {"ok": True}


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------


@router.post("/archive-expired")
async def archive_expired(request: Request) -> dict:
    body: dict = await request.json()
    count = await _svc.memory_archive_expired(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        batch_size=body.get("batch_size", 500),
    )
    return {"count": count}


@router.post("/archive-stale")
async def archive_stale_low_weight(request: Request) -> dict:
    body: dict = await request.json()
    count = await _svc.memory_archive_stale(
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        stale_days=body.get("stale_days", 90),
        max_weight=body.get("max_weight", 0.3),
        batch_size=body.get("batch_size", 500),
    )
    return {"count": count}


# ------------------------------------------------------------------
# Stats / analytics
# ------------------------------------------------------------------


@router.get("/stats")
async def get_memory_stats(
    tenant_id: str,
    fleet_id: str | None = None,
) -> dict:
    return await _svc.memory_compute_health_stats(tenant_id, fleet_id)


@router.get("/embedding-coverage")
async def get_embedding_coverage(
    tenant_id: str,
    fleet_id: str | None = None,
) -> dict:
    missing = await _svc.memory_find_missing_embeddings(tenant_id, fleet_id)
    total = await _svc.memory_count_active(tenant_id, fleet_id)
    return {
        "total_active": total,
        "missing_embeddings": len(missing),
        "coverage_pct": round((total - len(missing)) / total * 100, 1) if total > 0 else 0.0,
    }


@router.get("/type-distribution")
async def get_type_distribution(
    tenant_id: str,
    fleet_id: str | None = None,
) -> dict:
    stats = await _svc.memory_compute_health_stats(tenant_id, fleet_id)
    return {"type_distribution": stats.get("type_distribution", {})}


@router.get("/recent")
async def get_recent_memories(
    tenant_id: str,
    fleet_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    memories = await _svc.memory_list_recent(tenant_id, fleet_id, limit=limit)
    return [orm_to_dict(m, MEMORY_FIELDS) for m in memories]


@router.get("/lifecycle-candidates")
async def get_lifecycle_candidates(tenant_id: str) -> dict:
    expired = await _svc.memory_find_expired_still_active(tenant_id, None)
    stale = await _svc.memory_find_stale_count(tenant_id, None, stale_days=90, max_weight=0.3)
    return {
        "expired_still_active": [str(r[0]) for r in expired],
        "stale_low_weight": [str(r[0]) for r in stale],
    }


@router.get("/count")
async def count_memories(
    tenant_id: str,
    fleet_id: str | None = None,
) -> dict:
    if not tenant_id:
        count = await _svc.memory_count_all()
    else:
        count = await _svc.memory_count_active(tenant_id, fleet_id)
    return {"count": count}


@router.get("/count-active")
async def count_active_memories(tenant_id: str, fleet_id: str | None = None) -> dict:
    count = await _svc.memory_count_active(tenant_id, fleet_id)
    return {"count": count}


@router.get("/distinct-agents")
async def count_distinct_agents() -> dict:
    """Global count of distinct agent identities across all memories.

    Used by the public landing-page Agents counter.
    """
    count = await _svc.memory_distinct_agent_count()
    return {"count": count}


# ------------------------------------------------------------------
# Parameterised paths — MUST come last to avoid catching /count etc.
# ------------------------------------------------------------------


@router.get("/{memory_id}")
async def get_memory(memory_id: UUID, tenant_id: str | None = None) -> dict:
    t_start = time.perf_counter()
    db_timer = None
    memory = None
    success = True
    try:
        with bind_timer() as db_timer:
            if tenant_id is not None:
                memory = await _svc.memory_get_by_id_for_tenant(memory_id, tenant_id)
            else:
                memory = await _svc.memory_get_by_id(memory_id)
    except Exception:
        success = False
        raise
    finally:
        log_request(
            "memory-get",
            tenant_id=tenant_id,
            total_ms=(time.perf_counter() - t_start) * 1000,
            db_ms=db_timer.total_ms if db_timer is not None else 0.0,
            hit=memory is not None,
            error=not success,
        )
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return orm_to_dict(memory, MEMORY_FIELDS)


@router.patch("/{memory_id}")
async def update_memory(memory_id: UUID, request: Request) -> dict:
    body: dict = await request.json()
    # ``_parse_datetimes`` mirrors the POST route's ingress contract:
    # asyncpg requires datetime instances on ``DateTime(timezone=True)``
    # columns and rejects ISO strings with ``CannotCoerceError``. The
    # CAURA-595 async-enrich worker hits this path with bare ISO
    # strings via ``model_dump(mode="json")``; the POST route always
    # parsed but the PATCH route silently passed strings straight to
    # SQLAlchemy → asyncpg → 500. Parse here so all callers (worker,
    # core-api, future tooling) get the same coercion at the API
    # boundary.
    _parse_datetimes(body)
    if body:
        await _svc.memory_update(memory_id, body)
    return {"ok": True}


@router.patch("/{memory_id}/status")
async def update_memory_status(memory_id: UUID, request: Request) -> dict:
    body: dict = await request.json()
    status = body["status"]
    supersedes_id = body.get("supersedes_id")
    # Update status
    await _svc.memory_update_status(memory_id, status)
    # If supersedes_id is provided, also set that field via a direct update
    if supersedes_id is not None:
        from sqlalchemy import update as sql_update

        from common.models import Memory
        from core_storage_api.services.postgres_service import get_session

        async with get_session() as session:
            await session.execute(
                sql_update(Memory).where(Memory.id == memory_id).values(supersedes_id=UUID(supersedes_id))
            )
    return {"ok": True}


@router.patch("/{memory_id}/embedding")
async def update_embedding(memory_id: UUID, request: Request) -> dict:
    body: dict = await request.json()
    await _svc.memory_update_embedding(
        memory_id,
        embedding=body["embedding"],
        metadata=body.get("metadata"),
    )
    return {"ok": True}


@router.patch("/{memory_id}/entities")
async def update_memory_entities(memory_id: UUID, request: Request) -> dict:
    body: dict = await request.json()
    entity_links = body.get("entity_links", [])
    for link in entity_links:
        entity_id = UUID(link["entity_id"])
        role = link["role"]
        await _svc.memory_add_entity_link(memory_id, entity_id, role)
    return {"ok": True}


@router.delete("/{memory_id}")
async def soft_delete_memory(memory_id: UUID) -> dict:
    await _svc.memory_soft_delete(memory_id)
    return {"ok": True}
