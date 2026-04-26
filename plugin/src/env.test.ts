/**
 * Tests for tenant-id resolution behavior in env.ts.
 *
 * Guards the OSS-noise fix: when the backend is unreachable (undici throws
 * TypeError("fetch failed")), the resolver must short-circuit with one warn
 * line instead of 4 retries over ~14s.
 */
import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";

// Set API key before importing env.ts so resolveTenantId won't early-exit.
// Must be MEMCLAW_*-prefixed — env.ts only loads those from .env.
process.env.MEMCLAW_API_KEY = "mc_test_key_for_env_tests";
// Clear tenant id so resolveTenantId actually attempts a fetch.
delete process.env.MEMCLAW_TENANT_ID;

const { resolveTenantId } = await import("./env.js");

interface MockCall {
  url: string;
  init?: RequestInit;
}

let originalFetch: typeof fetch;
let calls: MockCall[];
let warnLines: string[];
let errorLines: string[];
let originalWarn: typeof console.warn;
let originalError: typeof console.error;

function installConsoleCapture(): void {
  warnLines = [];
  errorLines = [];
  originalWarn = console.warn;
  originalError = console.error;
  console.warn = (...args: unknown[]) => {
    warnLines.push(args.map((a) => String(a)).join(" "));
  };
  console.error = (...args: unknown[]) => {
    errorLines.push(args.map((a) => String(a)).join(" "));
  };
}

function restoreConsole(): void {
  console.warn = originalWarn;
  console.error = originalError;
}

describe("resolveTenantId — network failure handling", () => {
  beforeEach(() => {
    originalFetch = globalThis.fetch;
    calls = [];
    installConsoleCapture();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    restoreConsole();
  });

  test("short-circuits on TypeError('fetch failed') — one log line, no retry backoff", async () => {
    globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
      calls.push({ url: String(input), init });
      // undici throws this shape for DNS/ECONNREFUSED/TLS failures
      throw new TypeError("fetch failed");
    }) as typeof fetch;

    const t0 = Date.now();
    const result = await resolveTenantId();
    const elapsed = Date.now() - t0;

    assert.equal(result, "", "returns empty string on failure");
    assert.equal(calls.length, 1, "should only attempt fetch once (no retries)");
    assert.ok(elapsed < 500, `should short-circuit fast, took ${elapsed}ms`);
    assert.equal(
      warnLines.length,
      1,
      `expected 1 warn line, got ${warnLines.length}: ${warnLines.join(" | ")}`,
    );
    assert.match(warnLines[0], /tenant_id resolution skipped/);
    assert.match(warnLines[0], /standalone mode/);
    assert.equal(errorLines.length, 0, "no error-level output for network failures");
  });

  test("non-TypeError errors still follow the retry path (preserves 5xx/timeout behavior)", async () => {
    // Use a non-TypeError Error to exercise the else branch. We stub
    // setTimeout to fire immediately, collapsing the 14s backoff into
    // zero real time so the retry loop completes in microseconds.
    const originalSetTimeout = globalThis.setTimeout;
    globalThis.setTimeout = ((cb: () => void) => {
      // Fire immediately — ignore the delay. Return a dummy timer handle.
      Promise.resolve().then(cb);
      return 0 as unknown as ReturnType<typeof originalSetTimeout>;
    }) as unknown as typeof setTimeout;

    let callCount = 0;
    globalThis.fetch = (async () => {
      callCount++;
      throw new Error("socket timeout"); // plain Error, not TypeError
    }) as typeof fetch;

    try {
      const result = await resolveTenantId();
      assert.equal(result, "", "returns empty after all retries exhausted");
      assert.equal(callCount, 4, `expected 4 attempts (initial + 3 retries), got ${callCount}`);
      assert.equal(
        warnLines.filter((l) => /attempt \d+\/4 failed: socket timeout/.test(l)).length,
        3,
        `expected 3 retry-warn lines (attempts 1-3), got: ${warnLines.join(" | ")}`,
      );
      assert.ok(
        errorLines.some((l) => /failed after 4 attempts: socket timeout/.test(l)),
        `expected final error line, got: ${errorLines.join(" | ")}`,
      );
    } finally {
      globalThis.setTimeout = originalSetTimeout;
    }
  });
});
