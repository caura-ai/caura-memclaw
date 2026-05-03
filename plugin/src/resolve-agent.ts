/**
 * Resolve agent identity from the best available source.
 *
 * Resolution order:
 *   1. Explicit field from caller (context.agentId, config.agentId)
 *   2. Session key parsing — "agent:AGENT_NAME:CHANNEL:TARGET"
 *   3. Config agent name (config.agentName, config.agent?.name)
 *   4. MEMCLAW_AGENT_ID env var
 *   5. ``main-${installId}`` — install-disambiguated default so two
 *      OpenClaw installs sharing one tenant don't merge their memories
 *      into a single ``(tenant_id, agent_id="main")`` row. Pre-Task6
 *      this was ``"unknown-agent"``, which is the same collision risk
 *      with worse semantics.
 */

import { MEMCLAW_AGENT_ID } from "./env.js";
import { getInstallId } from "./install-id.js";

export function resolveAgentId(
  ...sources: Array<Record<string, unknown> | undefined | null>
): string {
  for (const src of sources) {
    if (!src || typeof src !== "object") continue;

    if (src.agentId && typeof src.agentId === "string") return src.agentId;
    if (src.agent_id && typeof src.agent_id === "string") return src.agent_id;

    const agent = src.agent as Record<string, unknown> | undefined;
    if (agent?.id && typeof agent.id === "string") return agent.id;
    if (agent?.name && typeof agent.name === "string") return agent.name;

    if (src.sessionKey && typeof src.sessionKey === "string") {
      const parts = (src.sessionKey as string).split(":");
      if (parts.length >= 2 && parts[0] === "agent" && parts[1]) {
        return parts[1];
      }
    }

    if (src.agentName && typeof src.agentName === "string")
      return src.agentName as string;
  }

  if (MEMCLAW_AGENT_ID) {
    console.warn(
      "[memclaw] Agent ID resolved from MEMCLAW_AGENT_ID env var — consider passing agent_id explicitly",
    );
    return MEMCLAW_AGENT_ID;
  }

  // Per-install fallback. Was ``"unknown-agent"`` pre-Task6 — every
  // install collided on a single row.
  const fallback = `main-${getInstallId()}`;
  console.warn(
    `[memclaw] Could not resolve agent ID — using install-default '${fallback}'. ` +
      `Pass agent_id explicitly (or set MEMCLAW_AGENT_ID) for clarity.`,
  );
  return fallback;
}
