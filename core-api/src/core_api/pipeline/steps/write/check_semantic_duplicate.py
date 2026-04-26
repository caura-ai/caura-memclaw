"""CheckSemanticDuplicate — reject near-duplicates via pgvector cosine similarity."""

from __future__ import annotations

import logging
import time

from fastapi import HTTPException

from core_api.pipeline.context import PipelineContext
from core_api.pipeline.step import StepOutcome, StepResult
from core_api.services.memory_service import _find_semantic_duplicate

logger = logging.getLogger(__name__)


class CheckSemanticDuplicate:
    @property
    def name(self) -> str:
        return "check_semantic_duplicate"

    async def execute(self, ctx: PipelineContext) -> StepResult | None:
        data = ctx.data["input"]
        tenant_config = ctx.tenant_config
        embedding = ctx.data["embedding"]
        fields = ctx.data["memory_fields"]
        metadata = fields["metadata"]

        if not tenant_config.semantic_dedup_enabled or embedding is None:
            return StepResult(outcome=StepOutcome.SKIPPED)

        t_dedup = time.perf_counter()
        sem_dup = await _find_semantic_duplicate(
            ctx.require_db,
            data.tenant_id,
            data.fleet_id,
            embedding,
            visibility=data.visibility or "scope_team",
        )
        dedup_ms = round((time.perf_counter() - t_dedup) * 1000, 1)
        metadata["semantic_dedup_ms"] = dedup_ms

        if sem_dup:
            raise HTTPException(
                status_code=409,
                detail=f"Near-duplicate memory exists: {sem_dup['id'] if isinstance(sem_dup, dict) else sem_dup.id}",
            )
        return None
