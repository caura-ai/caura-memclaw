"""ScheduleBackgroundTasks — fire-and-forget entity extraction, contradiction detection, re-embed."""

from __future__ import annotations

import logging

from core_api.config import settings
from core_api.pipeline.context import PipelineContext
from core_api.pipeline.step import StepResult
from core_api.services.entity_extraction_worker import process_entity_extraction
from core_api.services.task_tracker import tracked_task
from core_api.tasks import track_task

logger = logging.getLogger(__name__)


class ScheduleBackgroundTasks:
    @property
    def name(self) -> str:
        return "schedule_background_tasks"

    async def execute(self, ctx: PipelineContext) -> StepResult | None:
        data = ctx.data["input"]
        tenant_config = ctx.tenant_config
        memory = ctx.data["memory"]
        embedding = ctx.data["embedding"]
        enrichment = ctx.data.get("enrichment")
        resolved_write_mode = ctx.data.get("resolved_write_mode")
        memory_id = memory["id"] if isinstance(memory, dict) else memory.id

        # Fast mode: defer enrichment + entity extraction + contradiction to background
        if resolved_write_mode == "fast":
            if tenant_config.enrichment_enabled:
                from core_api.services.memory_service import (
                    _agent_provided_enrichment_fields,
                    _schedule_enrich_or_inline,
                )

                track_task(
                    tracked_task(
                        _schedule_enrich_or_inline(
                            memory_id,
                            data.content,
                            data.tenant_id,
                            data.fleet_id,
                            data.agent_id,
                            tenant_config,
                            agent_provided_fields=_agent_provided_enrichment_fields(data),
                            reference_datetime=getattr(data, "reference_datetime", None),
                        ),
                        "background_enrichment",
                        memory_id,
                        data.tenant_id,
                    )
                )
            # CAURA-594: deferred-path or inline-failure backfill — the
            # shim publishes EMBED_REQUESTED in async mode, retries
            # in-process when embed_on_hot_path=True.
            if embedding is None:
                from core_api.services.memory_service import (
                    _schedule_embed_or_reembed,
                )

                track_task(
                    tracked_task(
                        _schedule_embed_or_reembed(
                            memory_id,
                            data.content,
                            data.tenant_id,
                            content_hash=ctx.data.get("content_hash"),
                        ),
                        "embed_or_publish",
                        memory_id,
                        data.tenant_id,
                    )
                )
            return None

        # Strong mode (or no mode set): today's behavior

        # CAURA-595: when ``enrich_on_hot_path=False`` the parallel
        # embed/enrich step skipped the LLM call by design and
        # ``enrichment`` is None. Publish ``ENRICH_REQUESTED`` so the
        # worker fills the row in the background. ``enrich_on_hot_path=True``
        # already ran enrichment inline upstream; nothing to schedule.
        if (
            enrichment is None
            and not settings.enrich_on_hot_path
            and tenant_config.enrichment_enabled
            and tenant_config.enrichment_provider != "none"
        ):
            from core_api.services.memory_service import (
                _agent_provided_enrichment_fields,
                _schedule_enrich_or_inline,
            )

            track_task(
                tracked_task(
                    _schedule_enrich_or_inline(
                        memory_id,
                        data.content,
                        data.tenant_id,
                        data.fleet_id,
                        data.agent_id,
                        tenant_config,
                        agent_provided_fields=_agent_provided_enrichment_fields(data),
                        reference_datetime=getattr(data, "reference_datetime", None),
                    ),
                    "enrich_or_publish",
                    memory_id,
                    data.tenant_id,
                )
            )

        # Entity extraction (fire-and-forget).
        #
        # CAURA-595 (shortcut form): entity extraction is "off the hot
        # path" in the sense that the request doesn't await it, but the
        # coroutine still runs in core-api's event loop. Under burst
        # write load, extraction LLM calls compete with live traffic
        # (scaling doc §10 bottleneck #7). Full worker-fleet migration
        # waits on CAURA-593 (Pub/Sub publisher/subscriber wiring) +
        # a new worker service that subscribes to
        # Topics.Pipeline.ENTITY_EXTRACT_REQUESTED.
        if tenant_config.entity_extraction_enabled:
            track_task(
                tracked_task(
                    process_entity_extraction(
                        memory_id,
                        data.tenant_id,
                        data.fleet_id,
                        data.agent_id,
                        data.content,
                        data.memory_type,
                    ),
                    "entity_extraction",
                    memory_id,
                    data.tenant_id,
                )
            )

        # CAURA-594: deferred-path or inline-failure backfill — the shim
        # publishes EMBED_REQUESTED in async mode, retries in-process
        # when embed_on_hot_path=True.
        if embedding is None:
            from core_api.services.memory_service import (
                _schedule_embed_or_reembed,
            )

            track_task(
                tracked_task(
                    _schedule_embed_or_reembed(
                        memory_id,
                        data.content,
                        data.tenant_id,
                        content_hash=ctx.data.get("content_hash"),
                    ),
                    "embed_or_publish",
                    memory_id,
                    data.tenant_id,
                )
            )
        else:
            # Contradiction detection (post-commit async)
            from core_api.services.contradiction_detector import (
                detect_contradictions_async,
            )

            track_task(
                tracked_task(
                    detect_contradictions_async(
                        memory_id,
                        data.tenant_id,
                        data.fleet_id,
                        data.content,
                        embedding,
                    ),
                    "contradiction_detection",
                    memory_id,
                    data.tenant_id,
                )
            )
        return None
