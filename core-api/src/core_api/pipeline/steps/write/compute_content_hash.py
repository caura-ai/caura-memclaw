"""ComputeContentHash — SHA256(tenant:fleet:content) + cached embedding lookup."""

from __future__ import annotations

import logging

from core_api.clients.storage_client import get_storage_client
from core_api.pipeline.context import PipelineContext
from core_api.pipeline.step import StepResult
from core_api.services.memory_service import _content_hash

logger = logging.getLogger(__name__)


class ComputeContentHash:
    @property
    def name(self) -> str:
        return "compute_content_hash"

    async def execute(self, ctx: PipelineContext) -> StepResult | None:
        data = ctx.data["input"]

        ch = _content_hash(data.tenant_id, data.fleet_id, data.content) if data.persist else None
        ctx.data["content_hash"] = ch

        # Check for existing embedding from duplicate content (saves LLM call)
        cached_embedding = None
        if ch:
            sc = get_storage_client()
            cached_embedding = await sc.find_embedding_by_content_hash(
                data.tenant_id,
                ch,
            )

        ctx.data["cached_embedding"] = cached_embedding
        return None
