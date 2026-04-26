/**
 * Validation helpers for MemClaw plugin security.
 *
 * Covers: UUID format, HTTPS enforcement, path containment,
 * HMAC command signature verification, and prompt length caps.
 */

import { createHmac, timingSafeEqual } from "crypto";
import { resolve } from "path";
import { realpathSync, existsSync } from "fs";

// --- UUID validation ---

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SAFE_ID_RE = /^[\w-]{1,128}$/;

export function isValidUUID(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

export function isValidSafeId(value: unknown): value is string {
  return typeof value === "string" && SAFE_ID_RE.test(value);
}

export function assertSafePathSegment(
  value: unknown,
  label: string,
): asserts value is string {
  if (!isValidSafeId(value)) {
    throw new Error(
      `${label} must be 1-128 alphanumeric/dash/underscore characters, got: ${String(value).slice(0, 40)}`,
    );
  }
}

// --- HTTPS enforcement ---

export function warnIfInsecureUrl(apiUrl: string, apiKey: string): void {
  if (apiKey && apiUrl.startsWith("http://")) {
    console.warn(
      "[memclaw] WARNING: MEMCLAW_API_KEY is set but MEMCLAW_API_URL uses plain HTTP. " +
        "API key will be transmitted in cleartext. Use https:// in production.",
    );
  }
}

// --- Path containment ---

export function isContainedPath(child: string, parent: string): boolean {
  try {
    const resolvedChild = existsSync(child) ? realpathSync(child) : resolve(child);
    const resolvedParent = existsSync(parent) ? realpathSync(parent) : resolve(parent);
    return (
      resolvedChild === resolvedParent ||
      resolvedChild.startsWith(resolvedParent + "/")
    );
  } catch {
    return false;
  }
}

// --- HMAC command signature verification ---

const COMMAND_SIGNATURE_MAX_AGE_MS = 120_000; // 2 minutes

export function verifyCommandSignature(
  cmd: { id: string; command: string; payload?: Record<string, unknown>; timestamp?: string; signature?: string },
  secretKey: string,
): { valid: boolean; reason?: string } {
  if (!secretKey) {
    // No key configured — development/keyless mode.
    // Allow unsigned commands through, but reject if a signature IS present
    // (something is trying to look authenticated when we can't verify it).
    if (cmd.signature) {
      return { valid: false, reason: "no_secret_configured_but_signature_present" };
    }
    console.warn(
      `[memclaw] WARNING: accepting unsigned command "${cmd.command}" — set MEMCLAW_API_KEY to enable signature verification.`,
    );
    return { valid: true, reason: "no_secret_configured" };
  }

  if (!cmd.signature) {
    return { valid: false, reason: "missing_signature" };
  }

  if (!cmd.timestamp) {
    return { valid: false, reason: "missing_timestamp" };
  }

  // Check timestamp freshness
  const cmdTime = new Date(cmd.timestamp).getTime();
  if (isNaN(cmdTime)) {
    return { valid: false, reason: "invalid_timestamp" };
  }
  const age = Math.abs(Date.now() - cmdTime);
  if (age > COMMAND_SIGNATURE_MAX_AGE_MS) {
    return { valid: false, reason: "expired_timestamp" };
  }

  // Verify HMAC: sign(id + command + timestamp + payload)
  const payloadStr = cmd.payload ? JSON.stringify(cmd.payload) : "";
  const hmacInput = `${cmd.id}:${cmd.command}:${cmd.timestamp}:${payloadStr}`;
  const expected = createHmac("sha256", secretKey)
    .update(hmacInput)
    .digest("hex");

  const sigBuf = Buffer.from(cmd.signature, "hex");
  const expectedBuf = Buffer.from(expected, "hex");
  if (sigBuf.length !== expectedBuf.length || !timingSafeEqual(sigBuf, expectedBuf)) {
    return { valid: false, reason: "invalid_signature" };
  }

  return { valid: true };
}

// --- Prompt length cap ---

export const MAX_EDUCATE_PROMPT_LENGTH = 65_536; // 64KB

export function assertPromptLength(prompt: string): void {
  if (prompt.length > MAX_EDUCATE_PROMPT_LENGTH) {
    throw new Error(
      `Prompt too large (${prompt.length} bytes, max ${MAX_EDUCATE_PROMPT_LENGTH})`,
    );
  }
}
