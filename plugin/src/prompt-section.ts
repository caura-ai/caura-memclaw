/**
 * MemClaw Memory Prompt Section Builder
 *
 * Produces the system-prompt fragment that teaches the LLM how to use
 * MemClaw for persistent recall.  Two consumers:
 *
 *   1. registerMemoryPromptSection  — native path (requires kind:"memory")
 *   2. before_prompt_build hook     — fallback for older OpenClaw versions
 *
 * Content is a condensed distillation of ya-skill.md (the 3 rules,
 * write-triggers, and agent-identity reminder).  Each rule is emitted
 * only when the tool it references is present in availableTools.
 */

import { MEMCLAW_TOOLS } from "./tools.js";

// ── Core builder ────────────────────────────────────────────────────

function buildRecallLines(availableTools: Set<string>): string[] {
  // Only emit guidance for MemClaw tools that are actually available.
  const present = MEMCLAW_TOOLS.filter((t) => availableTools.has(t));
  if (present.length === 0) return [];

  const hasRecall = availableTools.has("memclaw_recall");
  const hasWrite = availableTools.has("memclaw_write");
  const hasManage = availableTools.has("memclaw_manage");
  const hasDoc = availableTools.has("memclaw_doc");
  const hasEvolve = availableTools.has("memclaw_evolve");

  const lines: string[] = [];

  lines.push("## MemClaw Memory");
  lines.push("");
  lines.push(
    "You have access to MemClaw, a persistent long-term memory system. " +
      "Use it every session. Three mandatory rules:",
  );
  lines.push("");

  // Rule 1 — Search before you start
  if (hasRecall) {
    lines.push(
      "**Rule 1 — Search before you start.** Always call `memclaw_recall` " +
        "at the beginning of a task to check what is already known. " +
        "Pass `include_brief=true` when you want an LLM-summarized context " +
        "paragraph instead of raw results. Never start cold.",
    );
  }

  // Rule 2 — Write when something matters
  if (hasWrite) {
    lines.push(
      "**Rule 2 — Write when something matters.** After completing work " +
        "or when something important happens, call `memclaw_write` with a " +
        "descriptive record including dates, names, paths, and outcomes. " +
        "For tasks longer than 30 minutes, write a checkpoint every " +
        "30 minutes.",
    );
  }

  // Rule 3 — Update when facts change
  if (hasWrite && hasRecall && hasManage) {
    lines.push(
      "**Rule 3 — Update when facts change.** Write the new fact, recall " +
        "the old memory with `memclaw_recall`, then mark it outdated with " +
        '`memclaw_manage(op="transition", memory_id=old_id, status="outdated")`.',
    );
  }

  lines.push("");

  // Condensed write triggers
  if (hasWrite) {
    lines.push(
      "**Write triggers** (non-negotiable): task completed, bug found/fixed, " +
        "deployment made, decision made, API discovered/changed, person " +
        "context updated, blocker found, commitment made, configuration " +
        "changed, error pattern identified. When in doubt, write it.",
    );
    lines.push("");
  }

  // Document Store
  if (hasDoc) {
    lines.push(
      "**Document Store**: Use `memclaw_doc` (op-dispatched: `write`, " +
        "`read`, `query`, `delete`) for structured data — customer records, " +
        "config, inventory, task lists. Documents live in named collections " +
        "with exact-match lookups — use memories (`memclaw_write`) for " +
        "unstructured knowledge that needs semantic search.",
    );
    lines.push("");
  }

  // Delete / transition guidance
  if (hasManage) {
    lines.push(
      "**Deleting & transitions**: Use `memclaw_manage(op=\"delete\")` only " +
        "when a memory is wrong or must be permanently removed. Prefer " +
        "`memclaw_manage(op=\"transition\", status=\"outdated\"|\"archived\")` " +
        "to supersede memories without losing history.",
    );
    lines.push("");
  }

  // Karpathy Loop — report outcomes
  if (hasEvolve) {
    lines.push(
      "**Report outcomes**: After acting on recalled memories, call " +
        "`memclaw_evolve` with `outcome_type=\"success\"|\"failure\"|\"partial\"` " +
        "and `related_ids=[...]`. Successes reinforce weights; failures " +
        "auto-generate preventive rule memories.",
    );
    lines.push("");
  }

  // Agent identity reminder
  lines.push(
    "**Identity**: Every MemClaw call MUST include your `agent_id`. " +
      "Do not rely on the default.",
  );

  // Available tools summary
  lines.push("");
  lines.push(`Available MemClaw tools: ${present.join(", ")}.`);
  lines.push("");

  return lines;
}

// ── Exports ─────────────────────────────────────────────────────────

/**
 * MemoryPromptSectionBuilder-compatible function.
 *
 * Signature matches OpenClaw's expected:
 *   ({ availableTools: Set<string>, citationsMode?: string }) => string[]
 */
export function memclawPromptSectionBuilder(params: {
  availableTools: Set<string>;
  citationsMode?: string;
}): string[] {
  return buildRecallLines(params.availableTools);
}

/**
 * Flatten the prompt section into a single string for use as
 * `prependSystemContext` in the before_prompt_build fallback path.
 */
export function memclawPromptSectionText(
  availableTools: Set<string>,
): string {
  return buildRecallLines(availableTools).join("\n");
}
