"""Document Store — structured JSONB records for agents."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core_api.auth import AuthContext, get_auth_context
from core_api.clients.storage_client import get_storage_client
from core_api.db.session import get_db
from core_api.middleware.idempotency import IDEMPOTENCY_HEADER, idempotency_for
from core_api.middleware.rate_limit import write_limit
from core_api.services.audit_service import log_action
from core_api.services.usage_service import check_and_increment_by_tenant as check_and_increment

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Store"])

# ── Schemas ──


class DocWriteRequest(BaseModel):
    tenant_id: str
    fleet_id: str | None = None
    collection: str = Field(min_length=1, max_length=200)
    doc_id: str = Field(min_length=1, max_length=500)
    data: dict


class DocQueryRequest(BaseModel):
    tenant_id: str
    fleet_id: str | None = None
    collection: str = Field(min_length=1, max_length=200)
    where: dict = Field(default_factory=dict)
    order_by: str | None = None
    order: str = Field(default="asc", pattern=r"^(asc|desc)$")
    limit: int = Field(default=20, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class DocOut(BaseModel):
    id: str
    tenant_id: str
    fleet_id: str | None
    collection: str
    doc_id: str
    data: dict
    created_at: datetime
    updated_at: datetime


# ── Helpers ──


def _dict_to_out(d: dict) -> DocOut:
    return DocOut(
        id=str(d.get("id", "")),
        tenant_id=d.get("tenant_id", ""),
        fleet_id=d.get("fleet_id"),
        collection=d.get("collection", ""),
        doc_id=d.get("doc_id", ""),
        data=d.get("data", {}),
        created_at=d.get("created_at", datetime.min),
        updated_at=d.get("updated_at", datetime.min),
    )


# ── Routes ──


@router.post("/documents", response_model=DocOut)
@write_limit
async def upsert_document(
    request: Request,
    body: DocWriteRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(None, alias=IDEMPOTENCY_HEADER),
):
    """Upsert a document. If collection+doc_id exists, data is replaced."""
    auth.enforce_tenant(body.tenant_id)
    auth.enforce_read_only()
    auth.enforce_usage_limits()
    _idem = await idempotency_for(request, body.tenant_id, idempotency_key)
    if _idem and (_replay := _idem.cached_replay):
        _body, _status = _replay
        return JSONResponse(content=_body, status_code=_status)
    if auth.tenant_id:
        await check_and_increment(db, body.tenant_id, "write")

    sc = get_storage_client()
    doc = await sc.upsert_document(
        {
            "tenant_id": body.tenant_id,
            "fleet_id": body.fleet_id,
            "collection": body.collection,
            "doc_id": body.doc_id,
            "data": body.data,
        }
    )
    if doc is None:
        raise HTTPException(status_code=500, detail="Document upsert returned no rows")
    await log_action(
        db,
        tenant_id=body.tenant_id,
        action="doc_upsert",
        resource_type="document",
        resource_id=doc.get("id"),
        detail={"collection": body.collection, "doc_id": body.doc_id},
    )
    await db.commit()
    out = _dict_to_out(doc)
    if _idem:
        await _idem.record(out.model_dump(mode="json"), 200)
    return out


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    tenant_id: str = Query(...),
    collection: str = Query(...),
    auth: AuthContext = Depends(get_auth_context),
):
    """Get a single document by collection + doc_id."""
    auth.enforce_tenant(tenant_id)
    sc = get_storage_client()
    doc = await sc.get_document(tenant_id=tenant_id, collection=collection, doc_id=doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _dict_to_out(doc)


@router.post("/documents/query")
async def query_documents(
    body: DocQueryRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    """Query documents by field equality filters on JSONB data."""
    auth.enforce_tenant(body.tenant_id)

    sc = get_storage_client()
    docs = await sc.query_documents(
        {
            "tenant_id": body.tenant_id,
            "collection": body.collection,
            "fleet_id": body.fleet_id,
            "where": body.where,
            "order_by": body.order_by,
            "order": body.order,
            "limit": body.limit,
            "offset": body.offset,
        }
    )

    return [_dict_to_out(d) for d in docs]


@router.get("/documents")
async def list_documents(
    tenant_id: str = Query(...),
    collection: str = Query(...),
    fleet_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(get_auth_context),
):
    """List all documents in a collection."""
    auth.enforce_tenant(tenant_id)
    sc = get_storage_client()
    docs = await sc.list_documents(
        tenant_id=tenant_id, collection=collection, fleet_id=fleet_id, limit=limit, offset=offset
    )
    return [_dict_to_out(d) for d in docs]


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    tenant_id: str = Query(...),
    collection: str = Query(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document by collection + doc_id."""
    auth.enforce_tenant(tenant_id)
    auth.enforce_read_only()
    sc = get_storage_client()
    deleted = await sc.delete_document(tenant_id=tenant_id, collection=collection, doc_id=doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    await log_action(
        db,
        tenant_id=tenant_id,
        action="doc_delete",
        resource_type="document",
        detail={"collection": collection, "doc_id": doc_id},
    )
    await db.commit()
