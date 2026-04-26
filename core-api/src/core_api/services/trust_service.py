"""Agent trust-level enforcement.

Shared by the MCP handlers in ``core_api.mcp_server`` and the REST routes
under ``core_api.routes`` so both surfaces gate on the same rule without
reaching into each other's private symbols.

``require_trust`` returns ``(trust, not_found, None)`` on pass and
``(trust, not_found, error_str)`` on fail. ``not_found`` is a typed flag
for the "agent row is missing" case so callers can distinguish "unknown
id" from "known id with insufficient trust" without parsing the error
string. The error string carries the ``_MCP_ERROR_PREFIX`` constant so
the MCP handler convention stays in one place; ``parse_trust_error``
strips that prefix so REST routes can surface the bare detail through
an ``HTTPException``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

# Keep the prefix in one place so ``parse_trust_error`` and ``require_trust``
# can't drift if someone edits one. ``str.removeprefix`` is a no-op when the
# prefix isn't present, so parse_trust_error stays robust against future
# error strings that happen not to start with it.
_MCP_ERROR_PREFIX = "Error (403): "


async def require_trust(
    db: AsyncSession,
    tenant_id: str,
    agent_id: str,
    min_level: int,
) -> tuple[int, bool, str | None]:
    """Look up the agent's trust level and gate on ``min_level``.

    Returns ``(trust, not_found, None)`` on pass, ``(trust, not_found,
    error_str)`` on fail. ``not_found`` is ``True`` when no agent row
    exists for ``(tenant_id, agent_id)``; missing agent still counts as
    trust 0 but the flag lets callers surface a clearer "not registered"
    error without parsing the string. Error format matches the wider MCP
    handler convention of ``"Error (403): …"``.
    """
    from core_api.services.agent_service import lookup_agent

    agent = await lookup_agent(db, tenant_id, agent_id)
    if agent is None:
        return (
            0,
            True,
            f"{_MCP_ERROR_PREFIX}Agent '{agent_id}' not found (trust_level=0 < required {min_level}).",
        )
    trust = agent.get("trust_level", 0) if isinstance(agent, dict) else getattr(agent, "trust_level", 0)
    if trust < min_level:
        return (
            trust,
            False,
            f"{_MCP_ERROR_PREFIX}Agent '{agent_id}' (trust_level={trust}) < required {min_level}.",
        )
    return trust, False, None


def parse_trust_error(terr: str) -> str:
    """Strip the ``_MCP_ERROR_PREFIX`` from an error produced by ``require_trust``.

    REST routes use this to surface the underlying reason as the detail
    of an ``HTTPException`` without leaking the MCP-style prefix. Uses
    ``str.removeprefix`` which is a no-op when the prefix isn't present —
    keeps the helper safe against future callers that feed in arbitrary
    error strings.
    """
    return terr.removeprefix(_MCP_ERROR_PREFIX)
