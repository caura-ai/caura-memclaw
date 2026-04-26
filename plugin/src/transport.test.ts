/**
 * Tests for apiCall in transport.ts.
 *
 * Guards the API-prefix consolidation: all resource paths must be
 * auto-prepended with MEMCLAW_API_PREFIX, and absolute "/api/..." paths
 * must be rejected so regressions surface at test time.
 */
import { test, describe, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";

process.env.MEMCLAW_API_KEY = "mc_test_key_for_transport_tests";
process.env.MEMCLAW_API_URL = "http://localhost:8000";
process.env.MEMCLAW_TENANT_ID = "t_test";

const { apiCall } = await import("./transport.js");

interface MockCall {
  url: string;
  init?: RequestInit;
}

let originalFetch: typeof fetch;
let calls: MockCall[];

function installOkFetch(): void {
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
}

describe("apiCall — MEMCLAW_API_PREFIX handling", () => {
  beforeEach(() => {
    originalFetch = globalThis.fetch;
    calls = [];
    installOkFetch();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  test("prepends MEMCLAW_API_PREFIX to resource paths", async () => {
    await apiCall("POST", "/search", { q: "x" });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://localhost:8000/api/v1/search");
  });

  test("rejects paths starting with MEMCLAW_API_PREFIX", async () => {
    await assert.rejects(
      () => apiCall("POST", "/api/v1/search", {}),
      /apiCall path must be a resource path/,
    );
    assert.equal(calls.length, 0, "should not reach fetch");
  });

  test("rejects paths with prefix but no leading slash", async () => {
    await assert.rejects(
      () => apiCall("POST", "api/v1/search", {}),
      /apiCall path must be a resource path/,
    );
    assert.equal(calls.length, 0, "should not reach fetch");
  });

  test("normalizes missing leading slash", async () => {
    await apiCall("GET", "memories");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://localhost:8000/api/v1/memories");
  });

  test("query params survive prefix prepend", async () => {
    await apiCall("GET", "/memories", undefined, { tenant_id: "t1" });
    assert.equal(calls.length, 1);
    assert.equal(
      calls[0].url,
      "http://localhost:8000/api/v1/memories?tenant_id=t1",
    );
  });
});
