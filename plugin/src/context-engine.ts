/**
 * MemClaw ContextEngine — lifecycle hooks for OpenClaw memory integration.
 *
 * Provides: bootstrap (smoke test), ingest (message buffering + persistence),
 * assemble (token-budget-aware recall injection), afterTurn (auto-write),
 * compact (persist compaction summaries).
 *
 * Security:
 * - afterTurn enabled by default; opt out with MEMCLAW_AUTO_WRITE_TURNS=false
 * - Recall timeout enforced via AbortController
 */

import { apiCall } from "./transport.js";
import {
  MEMCLAW_FLEET_ID,
  MEMCLAW_TENANT_ID,
  MEMCLAW_AUTO_WRITE_TURNS,
  ensureTenantId,
  RECALL_CACHE_TTL_MS,
  RECALL_TIMEOUT_MS,
  MIN_TURN_CONTENT_LENGTH,
  MAX_TURN_SUMMARY_LENGTH,
  MAX_RECALL_CONTENT_LENGTH,
} from "./env.js";
import { memclawPromptSectionText } from "./prompt-section.js";
import { MEMCLAW_TOOLS } from "./tools.js";
import { resolveAgentId } from "./resolve-agent.js";
import { logError, logErrorCritical } from "./logger.js";

// --- Typed interfaces for ContextEngine hooks ---

export interface IngestMessage {
  role: "user" | "assistant" | "system";
  content: string | unknown;
  sessionKey?: string;
}

export interface AssembleBudget {
  tokenBudget?: number;
}

export interface CompactContext {
  summary?: string;
  compactionSummary?: string;
  [key: string]: unknown;
}

export interface AfterTurnContext {
  messages?: Array<{ role: string; content: string | unknown }>;
  [key: string]: unknown;
}

// --- Session message buffer (LRU, per-session) ---

const SESSION_BUFFER_CAP = 50;
const MAX_SESSIONS = 100;
const MAX_INGEST_WRITES_PER_SESSION = 10;
const sessionBuffers = new Map<string, IngestMessage[]>();
const sessionIngestCounts = new Map<string, number>();

function getTenantPrefix(config: Record<string, unknown>): string {
  return (config.tenantId as string) || MEMCLAW_TENANT_ID || "default";
}

function getSessionKey(config: Record<string, unknown>): string {
  const tenantPrefix = getTenantPrefix(config);
  // Always prefix with tenant to prevent cross-tenant buffer sharing,
  // even when config.sessionKey is provided.
  const sessionPart =
    (config.sessionKey as string) ||
    resolveAgentId(config) + ":" + (config.sessionId || "default");
  return tenantPrefix + ":" + sessionPart;
}

function pushToBuffer(sessionKey: string, message: IngestMessage): void {
  let buffer = sessionBuffers.get(sessionKey);
  if (!buffer) {
    // LRU eviction: if we have too many sessions, drop the oldest
    if (sessionBuffers.size >= MAX_SESSIONS) {
      const oldest = sessionBuffers.keys().next().value!;
      sessionBuffers.delete(oldest);
      sessionIngestCounts.delete(oldest);
    }
    buffer = [];
    sessionBuffers.set(sessionKey, buffer);
  }
  buffer.push(message);
  // Cap per-session buffer
  if (buffer.length > SESSION_BUFFER_CAP) {
    buffer.splice(0, buffer.length - SESSION_BUFFER_CAP);
  }
}

// --- Build search query from recent user messages ---

function buildQueryFromMessages(
  sessionKey: string,
  fallbackPrompt?: string,
): string {
  const buffer = sessionBuffers.get(sessionKey);
  if (buffer && buffer.length > 0) {
    // Use last 3 user messages to build a contextual query
    const userMessages = buffer
      .filter((m) => m.role === "user")
      .slice(-3);
    if (userMessages.length > 0) {
      const combined = userMessages
        .map((m) =>
          typeof m.content === "string" ? m.content : JSON.stringify(m.content),
        )
        .join(" ");
      // Truncate to a reasonable query length
      return combined.length > 500 ? combined.slice(-500) : combined;
    }
  }
  return fallbackPrompt && fallbackPrompt.length > 5 ? fallbackPrompt : "";
}

// --- Token budget helpers ---

const CHARS_PER_TOKEN_ESTIMATE = 4;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
}

function trimToTokenBudget(text: string, maxTokens: number): string {
  const maxChars = maxTokens * CHARS_PER_TOKEN_ESTIMATE;
  if (text.length <= maxChars) return text;
  // Trim from the end, keeping complete lines where possible
  const trimmed = text.slice(0, maxChars);
  const lastNewline = trimmed.lastIndexOf("\n");
  return lastNewline > maxChars * 0.5
    ? trimmed.slice(0, lastNewline)
    : trimmed;
}

// --- Recall cache ---

const RECALL_CACHE_MAX_ENTRIES = 200;
const recallCache = new Map<string, { text: string; ts: number }>();

// --- ContextEngine class ---

export class MemClawContextEngine {
  private config: Record<string, unknown>;
  private _bootstrapped = false;
  private _bootstrapPromise: Promise<void> | null = null;

  /** Engine metadata — tells OpenClaw this engine owns compaction. */
  readonly info = {
    id: "memclaw",
    name: "MemClaw Context Engine",
    ownsCompaction: true,
  };

  constructor(config: Record<string, unknown>) {
    this.config = config;
  }

  async bootstrap(): Promise<void> {
    if (this._bootstrapped) return;
    if (!this._bootstrapPromise) {
      this._bootstrapPromise = this._doBootstrap().catch((e) => {
        this._bootstrapPromise = null;
        throw e;
      });
    }
    return this._bootstrapPromise;
  }

  private async _doBootstrap(): Promise<void> {
    const bootAgentId = resolveAgentId(this.config);
    console.log(
      `[memclaw] ContextEngine bootstrap: agent=${bootAgentId}, ` +
        `fleet=${MEMCLAW_FLEET_ID || "(unset)"}, ` +
        `config keys=${Object.keys(this.config || {}).join(",") || "(empty)"}`,
    );

    const testContent = `memclaw-smoke-${Date.now()}`;
    let writtenId: string | null = null;
    try {
      const tid = await ensureTenantId();
      const wr = (await apiCall("POST", "/memories", {
        tenant_id: tid,
        agent_id: "__health_check__",
        content: testContent,
        memory_type: "fact",
        tags: ["__smoke_test__"],
      })) as Record<string, unknown>;
      writtenId =
        (wr?.id as string) ||
        ((wr?.memory as Record<string, unknown>)?.id as string) ||
        ((wr?.data as Record<string, unknown>)?.id as string) ||
        null;
      if (!writtenId) {
        console.warn("[memclaw] bootstrap: could not extract memory ID — smoke test memory may not be cleaned up");
      }

      let top: Record<string, unknown> | undefined;
      let score = 0;
      for (let attempt = 0; attempt < 3; attempt++) {
        await new Promise((r) => setTimeout(r, 500));
        const sr = (await apiCall("POST", "/search", {
          tenant_id: tid,
          query: testContent,
          top_k: 1,
        })) as Record<string, unknown> | Record<string, unknown>[];
        const firstResult = Array.isArray(sr)
          ? sr[0]
          : ((sr?.results as Record<string, unknown>[]) || [])[0];
        top = firstResult as Record<string, unknown> | undefined;
        score = (top?.score as number) ?? (top?.similarity as number) ?? 0;
        if (top && score >= 0.7) break;
      }

      if (!top) {
        console.error(
          "[memclaw] SMOKE TEST FAILED: search returned no results — check EMBEDDING_PROVIDER",
        );
      } else if (score < 0.7) {
        console.error(
          `[memclaw] SMOKE TEST WARNING: score ${score.toFixed(3)} < 0.7 — embeddings may be degraded`,
        );
      } else {
        console.log(
          `[memclaw] Smoke test passed (score: ${score.toFixed(3)})`,
        );
      }
    } catch (e: unknown) {
      logErrorCritical("SMOKE TEST ERROR", e);
    } finally {
      if (writtenId) {
        apiCall(
          "DELETE",
          `/memories/${encodeURIComponent(writtenId)}`,
        ).catch(() => {});
      }
    }
    this._bootstrapped = true;
  }

  /**
   * ingest — buffer messages per session and persist user messages as episodes.
   * Enables buildQueryFromMessages for richer recall in assemble().
   */
  async ingest(message: IngestMessage): Promise<void> {
    await this.bootstrap();
    if (!message || !message.content) return;

    const sessionKey = getSessionKey(this.config);
    pushToBuffer(sessionKey, message);

    // Persist user messages as episode memories (async, non-blocking).
    // Capped at MAX_INGEST_WRITES_PER_SESSION to prevent memory spam in long sessions.
    // The in-memory buffer still receives all messages for buildQueryFromMessages.
    if (message.role === "user") {
      const content =
        typeof message.content === "string"
          ? message.content
          : JSON.stringify(message.content);
      if (content.length < MIN_TURN_CONTENT_LENGTH) return;

      const writeCount = sessionIngestCounts.get(sessionKey) || 0;
      if (writeCount >= MAX_INGEST_WRITES_PER_SESSION) return;

      try {
        const tid = await ensureTenantId();
        const agentId = resolveAgentId(this.config);
        const truncated =
          content.length > MAX_TURN_SUMMARY_LENGTH
            ? content.slice(0, MAX_TURN_SUMMARY_LENGTH) + "..."
            : content;
        await apiCall("POST", "/memories", {
          tenant_id: tid,
          agent_id: agentId,
          fleet_id: MEMCLAW_FLEET_ID || undefined,
          content: truncated,
          memory_type: "episode",
          tags: ["auto-ingest", "user-message"],
        });
        sessionIngestCounts.set(sessionKey, writeCount + 1);
      } catch (e: unknown) {
        logError("Failed to persist ingested message", e);
      }
    }
  }

  /**
   * assemble — called before every LLM call.
   * Respects tokenBudget with 20/80 split (education/recall).
   * Uses buildQueryFromMessages for better recall targeting.
   */
  async assemble(
    budget: AssembleBudget,
    prompt?: string,
  ): Promise<{
    system?: string;
    systemPromptAddition?: string;
    messages?: unknown[];
    tokenEstimate?: number;
    estimatedTokens?: number;
  }> {
    await this.bootstrap();
    const agentId = resolveAgentId(this.config);
    const fleetId = MEMCLAW_FLEET_ID || undefined;
    const tokenBudget = budget?.tokenBudget || 0;

    // --- Section 1: Education (usage rules + identity) ---
    const educationText = memclawPromptSectionText(new Set(MEMCLAW_TOOLS));
    const identityBlock =
      `\n**Your identity**: agent_id=\`${agentId}\`` +
      (fleetId ? `, fleet_id=\`${fleetId}\`` : "") +
      (MEMCLAW_TENANT_ID ? `, tenant_id=\`${MEMCLAW_TENANT_ID}\`` : "") +
      "\n";

    const operatorPrompt = process.env.MEMCLAW_EDUCATION_PROMPT || "";
    const operatorBlock = operatorPrompt
      ? `\n## Operator Instructions\n${operatorPrompt}\n`
      : "";

    const staticSection = educationText + identityBlock + operatorBlock;

    // --- Token budget split: 20% education, 80% recall ---
    let recallBudgetTokens = 0;
    if (tokenBudget > 0) {
      const staticTokens = estimateTokens(staticSection);
      const educationBudget = Math.floor(tokenBudget * 0.2);
      const recallBudget = tokenBudget - educationBudget;
      // If education exceeds its 20% budget, borrow from recall
      const educationOverflow = Math.max(0, staticTokens - educationBudget);
      recallBudgetTokens = Math.max(0, recallBudget - educationOverflow);
    }

    // --- Section 2: Recalled memories (cached) ---
    const sessionKey = getSessionKey(this.config);
    const queryFromMessages = buildQueryFromMessages(sessionKey, prompt);
    const searchQuery = queryFromMessages || prompt || agentId;
    const tenantPrefix = getTenantPrefix(this.config);
    const cacheKey = `${tenantPrefix}:${agentId}:${searchQuery}`;
    const cached = recallCache.get(cacheKey);
    let recallBlock = "";

    if (cached && Date.now() - cached.ts < RECALL_CACHE_TTL_MS) {
      recallBlock = cached.text;
    } else {
      // Evict stale entries
      const now = Date.now();
      for (const [k, v] of recallCache) {
        if (now - v.ts > RECALL_CACHE_TTL_MS) recallCache.delete(k);
      }

      const controller = new AbortController();
      const timeout = setTimeout(
        () => controller.abort(),
        RECALL_TIMEOUT_MS,
      );
      try {
        const tid = await ensureTenantId();

        const searchBody: Record<string, unknown> = {
          tenant_id: tid,
          filter_agent_id: agentId,
          query: searchQuery,
          top_k: 5,
        };

        const sr = (await apiCall(
          "POST",
          "/search",
          searchBody,
          undefined,
          controller.signal,
        )) as Record<string, unknown> | Record<string, unknown>[];

        const results = Array.isArray(sr)
          ? sr
          : ((sr as Record<string, unknown>)?.results as
              | Record<string, unknown>[]
              | undefined) || [];
        if (results.length > 0) {
          const lines = results.map(
            (m: Record<string, unknown>) =>
              `- [${(m.memory_type as string) || "memory"}] ${((m.content as string) || "").slice(0, MAX_RECALL_CONTENT_LENGTH)}`,
          );
          recallBlock =
            "\n## Recalled Memory Context\n" +
            "The following memories were retrieved from MemClaw for this session:\n" +
            lines.join("\n") +
            "\n";
        }
        if (recallBlock) {
          if (recallCache.size >= RECALL_CACHE_MAX_ENTRIES) {
            const oldest = recallCache.keys().next().value;
            if (oldest !== undefined) recallCache.delete(oldest);
          }
          recallCache.set(cacheKey, { text: recallBlock, ts: Date.now() });
        }
      } catch (e: unknown) {
        logError("recall failed", e);
      } finally {
        clearTimeout(timeout);
      }
    }

    // --- Apply token budget to recall if needed ---
    if (tokenBudget > 0) {
      if (recallBudgetTokens <= 0) {
        recallBlock = ""; // education alone fills the budget — drop recall
      } else if (recallBlock) {
        recallBlock = trimToTokenBudget(recallBlock, recallBudgetTokens);
      }
    }

    // --- Combine all sections ---
    const systemPromptAddition = staticSection + recallBlock;
    const estimatedTokens = estimateTokens(systemPromptAddition);

    if (!systemPromptAddition.trim()) return {};

    // Return both canonical (v2026.4.5+) and legacy (v2026.4.2) property names
    return tokenBudget > 0
      ? { system: systemPromptAddition, systemPromptAddition, tokenEstimate: estimatedTokens, estimatedTokens }
      : { system: systemPromptAddition, systemPromptAddition };
  }

  async compact(context: CompactContext): Promise<undefined> {
    const summary = context?.summary || context?.compactionSummary;
    if (summary && typeof summary === "string") {
      try {
        const tid = await ensureTenantId();
        const agentId = resolveAgentId(
          context as Record<string, unknown>,
          this.config,
        );
        await apiCall("POST", "/memories", {
          tenant_id: tid,
          agent_id: agentId,
          fleet_id: MEMCLAW_FLEET_ID || undefined,
          content: summary,
          memory_type: "episode",
          tags: ["auto-compaction"],
        });
      } catch (e: unknown) {
        logError("Failed to persist compaction summary", e);
      }
    }
    return undefined;
  }

  /** afterTurn — auto-write turn summary. Enabled by default; opt out with MEMCLAW_AUTO_WRITE_TURNS=false. */
  async afterTurn(context: AfterTurnContext): Promise<void> {
    if (!MEMCLAW_AUTO_WRITE_TURNS) return;

    const lastAssistant = context?.messages
      ?.filter((m) => m.role === "assistant")
      ?.slice(-1)?.[0];
    if (!lastAssistant?.content) return;

    const content =
      typeof lastAssistant.content === "string"
        ? lastAssistant.content
        : JSON.stringify(lastAssistant.content);
    if (content.length < MIN_TURN_CONTENT_LENGTH) return;

    try {
      const tid = await ensureTenantId();
      const agentId = resolveAgentId(
        context as Record<string, unknown>,
        this.config,
      );
      const turnSummary =
        content.length > MAX_TURN_SUMMARY_LENGTH
          ? content.slice(0, MAX_TURN_SUMMARY_LENGTH) + "..."
          : content;
      await apiCall("POST", "/memories", {
        tenant_id: tid,
        agent_id: agentId,
        fleet_id: MEMCLAW_FLEET_ID || undefined,
        content: turnSummary,
        memory_type: "episode",
        tags: ["auto-turn-summary"],
      });
    } catch (e: unknown) {
      logError("Failed to persist turn summary", e);
    }
  }

  async prepareSubagentSpawn(
    context: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return {
      memclawAgentId: resolveAgentId(context, this.config),
      memclawFleetId: MEMCLAW_FLEET_ID,
    };
  }

  async onSubagentEnded(_context: unknown): Promise<void> {}
}
