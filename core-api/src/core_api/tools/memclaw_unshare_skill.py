"""ToolSpec for memclaw_unshare_skill — remove a shared skill.

Default removes the skill from the catalog only — already-installed
nodes keep their local copy. Pass ``unshare_from_fleet=true`` (with
``target_fleet_id``) to also queue ``uninstall_skill`` fleet commands
that ``rm`` the local SKILL.md.
"""

from core_api import mcp_server

from ._builders import mcp_register
from ._registry import register
from ._types import ToolSpec

_DESCRIPTION = (
    "Remove a shared skill from the catalog. Default removes from catalog "
    "only; pass unshare_from_fleet=true (with target_fleet_id) to also "
    "delete the SKILL.md on every fleet node. Trust ≥ 1."
)

_SPEC = ToolSpec(
    name="memclaw_unshare_skill",
    description=_DESCRIPTION,
    handler=mcp_server.memclaw_unshare_skill,
    plugin_exposed=True,
    trust_required=1,
)
register(_SPEC)
mcp_register(mcp_server.mcp, _SPEC)
