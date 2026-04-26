"""Idempotency inbox — CRUD for the (tenant_id, key) → response cache."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core_storage_api.schemas import IDEMPOTENCY_RESPONSE_FIELDS, orm_to_dict
from core_storage_api.services.postgres_service import PostgresService

router = APIRouter(prefix="/idempotency", tags=["Idempotency"])
_svc = PostgresService()


class IdempotencyUpsertRequest(BaseModel):
    tenant_id: str
    idempotency_key: str
    request_hash: str
    response_body: dict
    status_code: int
    expires_at: datetime


@router.get("")
async def get_idempotency(tenant_id: str, idempotency_key: str) -> dict:
    row = await _svc.idempotency_get(
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="idempotency record not found or expired")
    return orm_to_dict(row, IDEMPOTENCY_RESPONSE_FIELDS)


@router.post("")
async def upsert_idempotency(body: IdempotencyUpsertRequest) -> dict:
    # On conflict, storage returns the existing row — the caller sees
    # whichever write won the race, not necessarily the one they sent.
    row = await _svc.idempotency_upsert(
        tenant_id=body.tenant_id,
        idempotency_key=body.idempotency_key,
        request_hash=body.request_hash,
        response_body=body.response_body,
        status_code=body.status_code,
        expires_at=body.expires_at,
    )
    return orm_to_dict(row, IDEMPOTENCY_RESPONSE_FIELDS)
