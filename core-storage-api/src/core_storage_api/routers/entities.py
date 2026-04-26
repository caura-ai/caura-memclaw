"""Entity CRUD, graph, relation, and memory-link endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from core_storage_api.schemas import (
    ENTITY_FIELDS,
    MEMORY_ENTITY_LINK_FIELDS,
    MEMORY_FIELDS,
    RELATION_FIELDS,
    orm_to_dict,
)
from core_storage_api.services.postgres_service import PostgresService

router = APIRouter(prefix="/entities", tags=["Entities"])
_svc = PostgresService()


# ------------------------------------------------------------------
# Entity CRUD (collection-level)
# ------------------------------------------------------------------


@router.post("")
async def create_entity(request: Request) -> dict:
    body: dict = await request.json()
    try:
        entity = await _svc.entity_add(body)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return orm_to_dict(entity, ENTITY_FIELDS)


@router.get("")
async def list_entities(
    tenant_id: str,
    fleet_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    entities = await _svc.entity_list(tenant_id, fleet_id=fleet_id, limit=limit, offset=offset)
    return [orm_to_dict(e, ENTITY_FIELDS) for e in entities]


@router.get("/exact")
async def find_exact_entity(
    tenant_id: str,
    name: str,
    entity_type: str = "default",
    fleet_id: str | None = None,
) -> dict:
    entity = await _svc.entity_find_exact(tenant_id, entity_type, name, fleet_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return orm_to_dict(entity, ENTITY_FIELDS)


# ------------------------------------------------------------------
# FTS
# ------------------------------------------------------------------


@router.post("/fts-search")
async def fts_search_entities(request: Request) -> list[str]:
    body: dict = await request.json()
    ids = await _svc.entity_fts_search(
        tokens=body["tokens"],
        tenant_id=body["tenant_id"],
        fleet_ids=body.get("fleet_ids"),
    )
    return [str(eid) for eid in ids]


# ------------------------------------------------------------------
# Embedding similarity (entity resolution)
# ------------------------------------------------------------------


@router.post("/embedding-similarity")
async def resolve_entity_candidates(request: Request) -> list[dict]:
    body: dict = await request.json()
    results = await _svc.entity_find_by_embedding_similarity(
        tenant_id=body["tenant_id"],
        entity_type=body["entity_type"],
        name_embedding=body["name_embedding"],
        fleet_id=body.get("fleet_id"),
        limit=body.get("limit", 5),
    )
    out = []
    for entity, sim in results:
        row = orm_to_dict(entity, ENTITY_FIELDS)
        row["similarity"] = float(sim)
        out.append(row)
    return out


# ------------------------------------------------------------------
# Graph
# ------------------------------------------------------------------


@router.post("/expand-graph")
async def expand_graph(request: Request) -> dict:
    body: dict = await request.json()
    result = await _svc.entity_expand_graph(
        seed_entity_ids=[UUID(eid) for eid in body["seed_entity_ids"]],
        tenant_id=body["tenant_id"],
        fleet_id=body.get("fleet_id"),
        max_hops=body.get("max_hops", 2),
        use_union=body.get("use_union", False),
    )
    return {str(eid): {"hop": hop, "weight": weight} for eid, (hop, weight) in result.items()}


@router.get("/full-graph")
async def get_full_graph(
    tenant_id: str,
    fleet_id: str | None = None,
) -> dict:
    entities, relations = await _svc.entity_get_full_graph(tenant_id, fleet_id)
    return {
        "entities": [orm_to_dict(e, ENTITY_FIELDS) for e in entities],
        "relations": [orm_to_dict(r, RELATION_FIELDS) for r in relations],
    }


# ------------------------------------------------------------------
# Relations
# ------------------------------------------------------------------


@router.post("/relations")
async def create_relation(request: Request) -> dict:
    body: dict = await request.json()
    relation = await _svc.relation_add(body)
    return orm_to_dict(relation, RELATION_FIELDS)


@router.get("/relations/find")
async def find_relation(
    source_id: str,
    target_id: str,
    relation_type: str,
) -> dict | None:
    # Derive tenant_id from source entity (client doesn't pass it).
    source = await _svc.entity_get_by_id(UUID(source_id))
    if source is None:
        raise HTTPException(status_code=404, detail="Source entity not found")
    relation = await _svc.relation_find(
        tenant_id=source.tenant_id,
        from_entity_id=UUID(source_id),
        relation_type=relation_type,
        to_entity_id=UUID(target_id),
    )
    if relation is None:
        raise HTTPException(status_code=404, detail="Relation not found")
    return orm_to_dict(relation, RELATION_FIELDS)


# ------------------------------------------------------------------
# Memory-entity links
# ------------------------------------------------------------------


@router.post("/links")
async def create_memory_entity_link(request: Request) -> dict:
    body: dict = await request.json()
    link = await _svc.entity_add_entity_link(body)
    return orm_to_dict(link, MEMORY_ENTITY_LINK_FIELDS)


@router.get("/links/find")
async def find_entity_link(
    memory_id: str,
    entity_id: str,
) -> dict | None:
    link = await _svc.entity_find_entity_link(
        memory_id=UUID(memory_id),
        entity_id=UUID(entity_id),
    )
    if link is None:
        raise HTTPException(status_code=404, detail="Link not found")
    return orm_to_dict(link, MEMORY_ENTITY_LINK_FIELDS)


@router.post("/memory-ids-by-entity-ids")
async def get_memory_ids_by_entity_ids(request: Request) -> list[dict]:
    body: dict = await request.json()
    entity_ids = [UUID(eid) for eid in body["entity_ids"]]
    links = await _svc.entity_get_memory_ids_by_entity_ids(entity_ids)
    return [{"memory_id": str(mid), "entity_id": str(eid), "role": role} for mid, eid, role in links]


@router.post("/count-memories")
async def count_memories_per_entity(request: Request) -> dict:
    body: dict = await request.json()
    entity_ids = [UUID(eid) for eid in body["entity_ids"]]
    counts = await _svc.entity_count_memories_per_entity(entity_ids)
    return {str(eid): count for eid, count in counts.items()}


# ------------------------------------------------------------------
# Crystallizer / health helpers
# ------------------------------------------------------------------


@router.get("/orphaned")
async def find_orphaned_entities(tenant_id: str) -> list[dict]:
    rows = await _svc.entity_find_orphaned(tenant_id, fleet_id=None)
    return [{"id": str(row[0]), "canonical_name": row[1]} for row in rows]


@router.get("/broken-links")
async def find_broken_entity_links(tenant_id: str) -> list[dict]:
    rows = await _svc.entity_find_broken_links(tenant_id, fleet_id=None)
    return [{"memory_id": str(row[0]), "entity_id": str(row[1])} for row in rows]


# ------------------------------------------------------------------
# Parameterised /{entity_id} routes — MUST stay at the bottom
# ------------------------------------------------------------------


@router.get("/{entity_id}")
async def get_entity(entity_id: UUID) -> dict:
    entity = await _svc.entity_get_by_id(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return orm_to_dict(entity, ENTITY_FIELDS)


@router.patch("/{entity_id}")
async def update_entity(entity_id: UUID, request: Request) -> dict:
    body: dict = await request.json()
    entity = await _svc.entity_update(entity_id, body)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return orm_to_dict(entity, ENTITY_FIELDS)


@router.get("/{entity_id}/with-memories")
async def get_entity_with_linked_memories(
    entity_id: UUID,
    tenant_id: str | None = None,
) -> dict:
    entity = await _svc.entity_get_by_id(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    t_id = tenant_id or entity.tenant_id
    rows = await _svc.entity_get_linked_memories(entity_id, t_id)
    return {
        "entity": orm_to_dict(entity, ENTITY_FIELDS),
        "linked_memories": [
            {
                "link": orm_to_dict(link, MEMORY_ENTITY_LINK_FIELDS),
                "memory": orm_to_dict(memory, MEMORY_FIELDS),
            }
            for link, memory in rows
        ],
    }


@router.get("/{entity_id}/relations")
async def get_outgoing_relations(
    entity_id: UUID,
    tenant_id: str | None = None,
) -> list[dict]:
    entity = await _svc.entity_get_by_id(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    t_id = tenant_id or entity.tenant_id
    rows = await _svc.relation_get_outgoing(entity_id, t_id)
    return [
        {
            "relation": orm_to_dict(rel, RELATION_FIELDS),
            "target": orm_to_dict(target, ENTITY_FIELDS),
        }
        for rel, target in rows
    ]
