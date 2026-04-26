import asyncio

from fastapi import APIRouter

from core_api.clients.storage_client import get_storage_client

router = APIRouter(tags=["System"])


@router.get("/stats")
async def public_stats():
    """Minimal public counters for the landing page status bar."""
    sc = get_storage_client()
    # Run both count queries concurrently — each is one round-trip to core-storage.
    memory_count, agent_count = await asyncio.gather(
        sc.count_all(tenant_id=""),
        sc.count_distinct_agents(),
    )
    return {
        "tenant_count": 1,
        "memory_count": memory_count,
        "agent_count": agent_count,
    }
