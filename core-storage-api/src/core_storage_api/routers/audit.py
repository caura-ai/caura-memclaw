"""Audit log endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

from core_storage_api.schemas import AUDIT_LOG_FIELDS, orm_to_dict
from core_storage_api.services.postgres_service import PostgresService

router = APIRouter(prefix="/audit-logs", tags=["Audit"])
_svc = PostgresService()


@router.post("")
async def create_audit_log(request: Request) -> dict:
    body: dict = await request.json()
    resource_id = body.get("resource_id")
    if resource_id is not None:
        resource_id = UUID(resource_id)
    await _svc.audit_add(
        tenant_id=body["tenant_id"],
        agent_id=body.get("agent_id"),
        action=body["action"],
        resource_type=body["resource_type"],
        resource_id=resource_id,
        detail=body.get("detail"),
    )
    return {"ok": True}


@router.get("")
async def list_audit_logs(
    tenant_id: str,
    limit: int = 50,
    offset: int = 0,
    action: str | None = None,
    resource_type: str | None = None,
) -> list[dict]:
    logs = await _svc.audit_list_by_tenant(tenant_id, limit=limit)
    results = [orm_to_dict(log, AUDIT_LOG_FIELDS) for log in logs]
    if action:
        results = [r for r in results if r.get("action") == action]
    if resource_type:
        results = [r for r in results if r.get("resource_type") == resource_type]
    if offset:
        results = results[offset:]
    return results
