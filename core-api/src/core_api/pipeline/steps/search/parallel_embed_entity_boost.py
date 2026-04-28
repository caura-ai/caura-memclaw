"""ParallelEmbedAndEntityBoost — concurrent embedding + entity/graph boost via asyncio.gather.

Uses the storage client for entity FTS and graph expansion instead of direct
repository calls.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import HTTPException

from core_api.clients.storage_client import get_storage_client
from core_api.constants import (
    GRAPH_HOP_BOOST,
    GRAPH_MAX_BOOSTED_MEMORIES,
    RECALL_BOOST_CAP,
)
from core_api.pipeline.context import PipelineContext
from core_api.pipeline.step import StepOutcome, StepResult
from core_api.services.entity_tokens import extract_entity_tokens
from core_api.services.memory_service import _get_or_cache_embedding

logger = logging.getLogger(__name__)


async def _entity_boost_via_storage(
    query: str,
    tenant_id: str,
    fleet_ids: list[str] | None,
    graph_expand: bool,
    graph_max_hops: int,
    use_union: bool = False,
    precomputed_hops: dict[UUID, tuple[int, float]] | None = None,
) -> tuple[set[UUID], dict[UUID, float]]:
    """Entity FTS → graph expansion → link collection via storage client.

    Returns (boosted_memory_ids, memory_boost_factor).
    """
    sc = get_storage_client()
    boosted_memory_ids: set[UUID] = set()
    memory_boost_factor: dict[UUID, float] = {}

    try:
        if precomputed_hops is not None:
            entity_hops = precomputed_hops
            matched_entity_ids = [eid for eid, (hop, _w) in entity_hops.items() if hop == 0]
        else:
            tokens = extract_entity_tokens(query)
            if not tokens:
                return boosted_memory_ids, memory_boost_factor

            fts_data: dict = {"tokens": tokens, "tenant_id": tenant_id}
            if fleet_ids:
                fts_data["fleet_ids"] = fleet_ids
            matched_id_strs = await sc.fts_search_entities(fts_data)
            matched_entity_ids = [UUID(eid) for eid in matched_id_strs]

            if not matched_entity_ids:
                return boosted_memory_ids, memory_boost_factor

            # Graph expansion
            if graph_expand and graph_max_hops > 0:
                expand_data = {
                    "seed_entity_ids": [str(eid) for eid in matched_entity_ids],
                    "tenant_id": tenant_id,
                    "fleet_id": fleet_ids[0] if fleet_ids and len(fleet_ids) == 1 else None,
                    "max_hops": graph_max_hops,
                    "use_union": use_union,
                }
                raw_hops = await sc.expand_graph(expand_data)
                entity_hops = {
                    UUID(eid_str): (hop_weight["hop"], hop_weight["weight"])
                    for eid_str, hop_weight in raw_hops.items()
                }
            else:
                entity_hops = dict.fromkeys(matched_entity_ids, (0, 1.0))

        if matched_entity_ids:
            all_entity_ids = list(entity_hops.keys())
            raw_links = await sc.get_memory_ids_by_entity_ids(
                [str(eid) for eid in all_entity_ids],
            )

            # Sort by hop order (closest entities first).
            all_links = sorted(
                raw_links,
                key=lambda row: entity_hops.get(UUID(row["entity_id"]), (999, 0.0))[0],
            )

            for link in all_links:
                mem_id_str, ent_id_str = link["memory_id"], link["entity_id"]
                mem_id = UUID(mem_id_str)
                ent_id = UUID(ent_id_str)
                if ent_id not in entity_hops:
                    continue
                hop, rel_weight = entity_hops[ent_id]
                hop_boost = GRAPH_HOP_BOOST.get(hop, GRAPH_HOP_BOOST[max(GRAPH_HOP_BOOST)])
                boost = hop_boost * rel_weight
                if mem_id not in memory_boost_factor or boost > memory_boost_factor[mem_id]:
                    memory_boost_factor[mem_id] = boost
                if len(memory_boost_factor) >= GRAPH_MAX_BOOSTED_MEMORIES:
                    break

            # Multi-entity boost
            if len(matched_entity_ids) > 1:
                memory_entity_count: dict[UUID, int] = {}
                matched_set = set(matched_entity_ids)
                for link in all_links:
                    mem_id = UUID(link["memory_id"])
                    ent_id = UUID(link["entity_id"])
                    if ent_id in matched_set:
                        memory_entity_count[mem_id] = memory_entity_count.get(mem_id, 0) + 1
                for mem_id in memory_boost_factor:
                    count = memory_entity_count.get(mem_id, 0)
                    if count > 1 and memory_boost_factor[mem_id] > 1.0:
                        extra = min(count - 1, 4) * 0.10
                        memory_boost_factor[mem_id] = min(
                            memory_boost_factor[mem_id] * (1.0 + extra),
                            RECALL_BOOST_CAP,
                        )

            boosted_memory_ids = set(memory_boost_factor.keys())
    except Exception:
        logger.exception("Entity/graph boost lookup failed (falling back to pure vector search)")

    return boosted_memory_ids, memory_boost_factor


class ParallelEmbedAndEntityBoost:
    @property
    def name(self) -> str:
        return "parallel_embed_entity_boost"

    async def execute(self, ctx: PipelineContext) -> StepResult | None:
        plan = ctx.data.get("retrieval_plan")
        if plan and plan.skip_embedding:
            return StepResult(outcome=StepOutcome.SKIPPED)

        data = ctx.data
        sp = data["search_params"]

        emb_task = asyncio.ensure_future(
            _get_or_cache_embedding(data["query"], data["tenant_id"], data["tenant_config"])
        )
        ent_task = asyncio.ensure_future(
            _entity_boost_via_storage(
                data["query"],
                data["tenant_id"],
                data.get("fleet_ids"),
                data.get("graph_expand", True),
                sp["graph_max_hops"],
                use_union=True,
                precomputed_hops=data.pop("_classified_entity_hops", None),
            )
        )
        try:
            embedding, (boosted_memory_ids, memory_boost_factor) = await asyncio.wait_for(
                asyncio.gather(emb_task, ent_task),
                timeout=15.0,
            )
        except TimeoutError:
            emb_task.cancel()
            ent_task.cancel()
            raise HTTPException(status_code=504, detail="Search embedding timed out")
        except ValueError as exc:
            emb_task.cancel()
            ent_task.cancel()
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception:
            emb_task.cancel()
            ent_task.cancel()
            raise

        data["embedding"] = embedding
        data["boosted_memory_ids"] = boosted_memory_ids
        data["memory_boost_factor"] = memory_boost_factor
        return None
