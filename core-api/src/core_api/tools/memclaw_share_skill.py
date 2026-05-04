"""ToolSpec for memclaw_share_skill — distribute a SKILL.md to fleet agents.

Stores the skill in the shared ``skills`` document collection and queues
``install_skill`` fleet commands per target node. Recipient plugins
materialise the skill into ``plugin/skills/<name>/SKILL.md`` on their
next heartbeat; OpenClaw's native skill discovery picks it up.
"""

from core_api import mcp_server

from ._builders import mcp_register
from ._registry import register
from ._types import ToolSpec

_DESCRIPTION = (
    "Share a SKILL.md with the fleet. Default publishes to the catalog "
    "(memclaw_doc op=query collection=skills); pass install_on_fleet=true "
    "to also auto-install on every node in target_fleet_id. Trust ≥ 1."
)

_SPEC = ToolSpec(
    name="memclaw_share_skill",
    description=_DESCRIPTION,
    handler=mcp_server.memclaw_share_skill,
    plugin_exposed=True,
    trust_required=1,
)
register(_SPEC)
mcp_register(mcp_server.mcp, _SPEC)
