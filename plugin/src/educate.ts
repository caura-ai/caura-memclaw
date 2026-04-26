/**
 * Agent education — write education prompts to HEARTBEAT.md in agent
 * workspaces, and append MemClaw sections to TOOLS.md and AGENTS.md.
 *
 * SKILL.md is no longer written per-workspace: it ships as a static file at
 * `<plugin-root>/skills/memclaw/SKILL.md` and is discovered by OpenClaw via
 * the `skills` field in `openclaw.plugin.json`.
 *
 * Security fixes:
 * - Path containment check prevents traversal attacks
 * - Prompt length cap prevents disk exhaustion
 */

import {
  readFileSync,
  writeFileSync,
  existsSync,
  readdirSync,
  statSync,
  unlinkSync,
} from "fs";
import { join, basename, resolve } from "path";
import { isContainedPath, assertPromptLength } from "./validation.js";
import { getOpenClawBaseDir } from "./paths.js";
import { MEMCLAW_TOOLS } from "./tools.js";
import { logError } from "./logger.js";

/**
 * Resolve all agent workspace directories using the canonical 4-source
 * discovery, deduplicated and existence-filtered:
 *
 *  1. <baseDir>/workspace                — default workspace, id="main"
 *  2. <baseDir>/workspace-<name>         — hyphen-prefix at baseDir, id=<name>
 *  3. agents.list[].workspace            — from openclaw.json (canonical),
 *                                          id from agent.id || agent.name
 *  4. <baseDir>/workspaces/<name>/       — subdir of plural parent,
 *                                          id=<name>; this is the path
 *                                          `openclaw agents add --workspace`
 *                                          places workspaces under by default.
 *
 * Path containment: every resolved path must live inside baseDir.
 * Dedup: identical resolved paths are returned once (first source wins).
 * Filter: when `agentIds` is non-empty, only entries whose id is in the
 *         set are returned.
 *
 * Replaces the previous `readdirSync(baseDir).filter(startsWith("workspace"))`
 * walk used by writeEducationFiles, which (a) missed agents under
 * `<baseDir>/workspaces/<name>/`, and (b) treated the literal `workspaces`
 * plural dir as a single workspace, producing phantom education files.
 */
export function discoverAgentWorkspaces(
  baseDir: string,
  agentIds?: string[],
): Array<{ path: string; id: string }> {
  const filterSet = agentIds?.length ? new Set(agentIds) : null;
  const seen = new Set<string>();
  const results: Array<{ path: string; id: string }> = [];

  function addWs(dir: string, id: string): void {
    if (!id) return;
    const resolved = dir.startsWith("/") ? resolve(dir) : resolve(join(baseDir, dir));
    if (!isContainedPath(resolved, baseDir)) {
      console.warn(
        `[memclaw] Rejected workspace path outside openclawDir: ${resolved}`,
      );
      return;
    }
    if (seen.has(resolved) || !existsSync(resolved)) return;
    try {
      if (!statSync(resolved).isDirectory()) return;
    } catch {
      return;
    }
    if (filterSet && !filterSet.has(id)) return;
    seen.add(resolved);
    results.push({ path: resolved, id });
  }

  // 1. Default workspace
  addWs(join(baseDir, "workspace"), "main");

  // 2. Hyphen-prefix at baseDir: <baseDir>/workspace-<name>
  try {
    for (const entry of readdirSync(baseDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      if (!entry.name.startsWith("workspace-")) continue;
      const id = entry.name.replace(/^workspace-/, "");
      addWs(join(baseDir, entry.name), id);
    }
  } catch (e: unknown) {
    logError("Failed to scan baseDir for hyphen-prefix workspaces", e);
  }

  // 3. agents.list[].workspace from openclaw.json
  try {
    const configPath = join(baseDir, "openclaw.json");
    if (existsSync(configPath)) {
      const config = JSON.parse(readFileSync(configPath, "utf-8"));
      const agentList = config?.agents?.list;
      if (Array.isArray(agentList)) {
        for (const agent of agentList) {
          if (agent.workspace) {
            addWs(agent.workspace, agent.id || agent.name);
          }
        }
      }
    }
  } catch (e: unknown) {
    logError("Failed to read agent config in discoverAgentWorkspaces", e);
  }

  // 4. <baseDir>/workspaces/<name>/ — used by `openclaw agents add`
  const wsParent = join(baseDir, "workspaces");
  if (existsSync(wsParent)) {
    try {
      if (statSync(wsParent).isDirectory()) {
        for (const entry of readdirSync(wsParent, { withFileTypes: true })) {
          if (!entry.isDirectory()) continue;
          addWs(join(wsParent, entry.name), entry.name);
        }
      }
    } catch (e: unknown) {
      logError("Failed to scan workspaces/ subdirectories", e);
    }
  }

  return results;
}

/**
 * Delete orphan TOOLS.md / AGENTS.md at <baseDir>/workspaces/ that the
 * pre-fix discovery wrote when it mistakenly treated the plural parent
 * directory as a single workspace. Idempotent: missing files are no-op,
 * non-MemClaw content at that path is left alone.
 */
function cleanupPhantomEducationFiles(baseDir: string): void {
  const phantomDir = join(baseDir, "workspaces");
  for (const fname of ["TOOLS.md", "AGENTS.md"]) {
    const fpath = join(phantomDir, fname);
    if (!existsSync(fpath)) continue;
    try {
      const content = readFileSync(fpath, "utf-8");
      const isMemClawOrphan =
        (fname === "TOOLS.md" && content.includes("MemClaw — Tools Available")) ||
        (fname === "AGENTS.md" && content.includes("## Memory V2"));
      if (isMemClawOrphan) {
        unlinkSync(fpath);
      }
    } catch (e: unknown) {
      logError(`Failed to clean phantom education file ${fpath}`, e);
    }
  }
}

export function educateAgents(
  prompt: string,
  agentIds?: string[],
  baseDir?: string,
): {
  count: number;
  educated: string[];
  failed: Array<{ workspace: string; error: string }>;
  verified: number;
} {
  assertPromptLength(prompt);

  const openclawDir = baseDir || getOpenClawBaseDir();
  const wsList = discoverAgentWorkspaces(openclawDir, agentIds);

  let count = 0;
  let verified = 0;
  const educated: string[] = [];
  const failed: Array<{ workspace: string; error: string }> = [];

  for (const { path: wsDir } of wsList) {
    const wsName = basename(wsDir);
    const hbPath = join(wsDir, "HEARTBEAT.md");
    try {
      const existing = existsSync(hbPath)
        ? readFileSync(hbPath, "utf-8").trim()
        : "";

      // Idempotency: skip if prompt is already present
      if (existing && existing.includes(prompt.trim())) {
        verified++;
        count++;
        educated.push(wsName);
        continue;
      }

      const newContent = existing
        ? existing + "\n\n---\n\n" + prompt.trim() + "\n"
        : prompt.trim() + "\n";

      // Size cap: prevent unbounded growth from repeated educate calls
      const MAX_HEARTBEAT_SIZE = 256 * 1024;
      if (newContent.length > MAX_HEARTBEAT_SIZE) {
        failed.push({ workspace: wsName, error: "HEARTBEAT.md would exceed 256KB limit" });
        continue;
      }

      writeFileSync(hbPath, newContent, "utf-8");

      const readBack = readFileSync(hbPath, "utf-8");
      if (readBack.includes(prompt.trim())) {
        verified++;
        count++;
        educated.push(wsName);
      } else {
        failed.push({
          workspace: wsName,
          error: "Write succeeded but verification failed",
        });
      }
    } catch (e: unknown) {
      const msg = logError(`educate write failed for ${wsName}`, e);
      failed.push({ workspace: wsName, error: msg });
    }
  }

  return { count, educated, failed, verified };
}

// --- Education file builders ---
//
// Content is intentionally transport-neutral: no host ("plugin" / "gateway" /
// host-product) references, no env-var interpolation, no version tag.
// These files are the canonical MemClaw agent education payload and may be
// loaded by host-managed workspaces or served to MCP/REST callers through
// other transports.
//
// Role separation — cost-aware split between per-turn and on-demand files.
//
//   SKILL.md   — static file, shipped at `<plugin-root>/skills/memclaw/`
//                via `openclaw.plugin.json:skills`. Loaded by the model
//                via the `read` tool ON DEMAND; only the skill-list entry
//                (name + description + path) appears in every turn. Owns
//                the deep reference: mental model, identity, three rules,
//                trust levels, sharing semantics, container choice,
//                quality, session loop, per-tool signatures, decision
//                guidance, constraints, and error codes.
//
//   TOOLS.md   — per-workspace append, INJECTED EVERY TURN as bootstrap
//                (subject to 12 K per-file / 60 K total char caps). Kept
//                lean on purpose: quick matrix (9 tools × purpose ×
//                returns) + enum vocabulary table + pointer to SKILL.md.
//                Retained because sub-agent sessions get only AGENTS.md
//                and TOOLS.md (other bootstrap files are filtered out),
//                so the enum vocab must be reachable without requiring
//                a SKILL.md read.
//
//   AGENTS.md  — per-workspace append, INJECTED EVERY TURN as bootstrap.
//                Behavioral enforcement: identity mandate, completion
//                contract, write triggers, capture cadences, quality
//                enforcement, prohibited behaviors. Short supersession
//                paragraph instructs the model to read SKILL.md before
//                its first MemClaw call.
//
// AGENTS.md idempotency is keyed off the substring "## Memory V2" in
// writeEducationFiles() below; buildAgentsMd() must continue to emit that
// exact substring.

export function buildToolsMd(): string {
  return `
---

## MemClaw — Tools Available

Persistent, cross-session, multi-agent memory. For per-tool signatures,
decision guidance, constraints, and error codes, read
\`skills/memclaw/SKILL.md\` before your first call in a session.

\`agent_id\` is resolved by your runtime — never fabricate.

### Quick matrix · ${MEMCLAW_TOOLS.length} tools

| Tool | Purpose | Returns |
|------|---------|---------|
| \`memclaw_recall\`     | Semantic + keyword search                           | \`[{id, content, score, memory_type, …}]\` |
| \`memclaw_write\`      | Store one (\`content\`) or batch ≤100 (\`items\`)       | \`{id}\` or \`{ids[]}\` |
| \`memclaw_manage\`     | Per-memory: read / update / transition / delete     | op-dispatched |
| \`memclaw_list\`       | Non-semantic browse (filter, sort, paginate)        | \`{results[], cursor}\` |
| \`memclaw_doc\`        | Structured-doc CRUD in named collections            | op-dispatched |
| \`memclaw_entity_get\` | Entity by UUID                                      | \`{entity}\` |
| \`memclaw_tune\`       | Update retrieval profile (sticky, not per-call)     | current profile |
| \`memclaw_insights\`   | Reflect: contradictions / failures / patterns / …   | stored as \`insight\` memories |
| \`memclaw_evolve\`     | Report outcome after acting on recalled memories    | weight updates; may create rules |

### Vocabulary

Enum values here mirror the JSON Schema in the registered tools; keep
in sync.

| Field | Valid values |
|-------|--------------|
| \`memory_type\` (auto on write; filter on read) | \`fact\`, \`episode\`, \`decision\`, \`preference\`, \`task\`, \`semantic\`, \`intention\`, \`plan\`, \`commitment\`, \`action\`, \`outcome\`, \`cancellation\`, \`rule\`, \`insight\` |
| \`status\` (via \`memclaw_manage op=transition\`) | \`active\`, \`pending\`, \`confirmed\`, \`cancelled\`, \`outdated\`, \`conflicted\`, \`archived\`, \`deleted\` |
| \`visibility\` (write-time) | \`scope_agent\` · \`scope_team\` *(default)* · \`scope_org\` |
| \`scope\` (read-time on \`_list\`, \`_insights\`) | \`agent\` *(default)* · \`fleet\` · \`all\` |
| \`fleet_ids\` (optional recall filter) | array of fleet ID strings; narrows recall to those fleets (trust 2 for cross-fleet) |
| \`write_mode\` | \`fast\` · \`auto\` *(default)* · \`strong\` |
| \`focus\` (\`_insights\`) | \`contradictions\` · \`failures\` · \`stale\` · \`divergence\` · \`patterns\` · \`discover\` |
| \`outcome_type\` (\`_evolve\`) | \`success\` · \`failure\` · \`partial\` |
`;
}

export function buildAgentsMd(): string {
  return `

---

## Memory V2 — MemClaw Protocol (mandatory)

Supersedes any earlier memory instructions in this file. MemClaw is the
primary persistent, cross-session, multi-agent memory; prior file-based
memory (e.g. \`memory.md\`) is session-local scratchpad only.

**Before your first MemClaw tool call in a session**, read
\`skills/memclaw/SKILL.md\` — it holds the per-tool signatures, decision
guidance, constraints, and error codes. \`TOOLS.md\` in this workspace
carries the at-a-glance tool list and enum vocabulary every turn.

### Identity is non-negotiable

- Every call MUST carry your correct \`agent_id\`. Never fabricate, reuse
  another agent's id, or hardcode a placeholder.
- \`fleet_id\` MUST be correct for team/org visibility writes,
  fleet-scoped reads, and cross-fleet operations.
- If uncertain, do NOT guess. Write privately
  (\`visibility=scope_agent\`) until identity is resolved.

### Completion contract

- No silent completions. Every completion MUST produce a write: what was
  done, found, changed, what's next.
- Long tasks (> 30 min) MUST checkpoint every 30 minutes.
- No write = not completed.

### Write triggers (non-negotiable)

Task completed · Bug found or fixed · Deployment performed · Decision made · API discovered or changed · Person context updated · Blocker encountered · Commitment made · Configuration changed · Error pattern identified. If in doubt: WRITE IT.

### 3-layer memory capture (mandatory)

- **L1 per-turn.** After each meaningful outcome, write with date,
  what, who, outcome, next.
- **L2 session boundary.** At > 60 % context or session end, write a
  full summary.
- **L3 consolidation.** On periodic runtime sweeps: find gaps, merge
  duplicates, transition contradicted facts to \`outdated\`.

### Orchestrator + subagent protocol

If your runtime dispatches subagents:

- The spawning agent MUST write findings after every subagent completion.
- The subagent MUST write its own findings before handing back.
- Both writes MUST carry their own \`agent_id\`.

Single-agent runtimes ignore this section.

### Quality enforcement

- Every memory MUST include a date.
- Prefer update over near-duplicate.
- Before a key fact, recall for contradictions; transition the older
  memory to \`outdated\` in the same turn.
- One topic per memory; batch into a single call.

### Prohibited behaviors

- NEVER fabricate or impersonate \`agent_id\` / \`fleet_id\`.
- NEVER delete memories you merely disagree with — transition them to
  \`outdated\` or \`archived\`. \`op=delete\` is a soft-delete; it requires
  trust 3 and is reserved for correcting genuinely wrong data.
- NEVER write with organization-wide visibility unless the memory is
  genuinely org-relevant.
- NEVER silently drop a denied call — surface the error so the
  orchestrator can decide whether to escalate.
- NEVER substitute local files or scratchpads for MemClaw writes.
`;
}

/**
 * Append MemClaw sections to TOOLS.md and AGENTS.md in each agent workspace.
 *
 * SKILL.md is no longer written here: it ships as a static file at
 * `<plugin-root>/skills/memclaw/SKILL.md` and is discovered by OpenClaw
 * via the `skills` field in `openclaw.plugin.json` — one copy per node,
 * auto-gated by plugin enablement.
 *
 * Called from both auto-education (first load) and the "educate" command.
 */
export function writeEducationFiles(
  toolsMdSection: string,
  agentsMdSection: string,
  agentIds?: string[],
  baseDir?: string,
): { toolsUpdated: number; agentsUpdated: number } {
  const ocBase = baseDir || getOpenClawBaseDir();
  const wsList = discoverAgentWorkspaces(ocBase, agentIds);
  let toolsUpdated = 0;
  let agentsUpdated = 0;

  for (const { path: wsPath } of wsList) {
    try {
      // TOOLS.md — append if not already present
      const toolsPath = join(wsPath, "TOOLS.md");
      const existingTools = existsSync(toolsPath) ? readFileSync(toolsPath, "utf-8") : "";
      if (!existingTools.includes("MemClaw")) {
        writeFileSync(toolsPath, existingTools + toolsMdSection, "utf-8");
        toolsUpdated++;
      }

      // AGENTS.md — append if not already present
      const agentsPath = join(wsPath, "AGENTS.md");
      const existingAgents = existsSync(agentsPath) ? readFileSync(agentsPath, "utf-8") : "";
      if (!existingAgents.includes("## Memory V2")) {
        writeFileSync(agentsPath, existingAgents + agentsMdSection, "utf-8");
        agentsUpdated++;
      }
    } catch {
      // Skip workspace on error
    }
  }

  // One-shot cleanup of phantom files written by the pre-fix discovery,
  // which mistakenly treated <baseDir>/workspaces/ (the plural parent
  // directory) as a single workspace. Idempotent.
  cleanupPhantomEducationFiles(ocBase);

  return { toolsUpdated, agentsUpdated };
}
