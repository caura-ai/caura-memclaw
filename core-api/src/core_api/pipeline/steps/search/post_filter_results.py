"""PostFilterResults — filter raw rows by min_similarity gate on vec_sim."""

from __future__ import annotations

from core_api.pipeline.context import PipelineContext
from core_api.pipeline.step import StepOutcome, StepResult
from core_api.pipeline.steps.search.retrieval_types import RetrievalStrategy


class PostFilterResults:
    @property
    def name(self) -> str:
        return "post_filter_results"

    async def execute(self, ctx: PipelineContext) -> StepResult | None:
        plan = ctx.data.get("retrieval_plan")
        if plan and plan.strategy == RetrievalStrategy.ENTITY_LOOKUP:
            return StepResult(outcome=StepOutcome.SKIPPED)

        min_similarity = ctx.data["search_params"]["min_similarity"]
        # Mirror the legacy path in memory_service.py: rows with vec_sim=None
        # are FTS-only matches that must not be gated by cosine similarity.
        # Currently unreachable in practice (the scored_search SQL requires
        # `embedding IS NOT NULL` and the storage layer coerces None → 0.0),
        # but kept aligned so the two search code paths never diverge if
        # either invariant changes to preserve FTS-only rows.
        filtered = [
            row for row in ctx.data["raw_rows"] if row.vec_sim is None or float(row.vec_sim) >= min_similarity
        ]
        # Trim to the user-requested top_k (storage returned top_k * overfetch_factor)
        final_top_k = ctx.data.get("final_top_k")
        if final_top_k is not None:
            filtered = filtered[:final_top_k]
        ctx.data["filtered_rows"] = filtered
        return None
