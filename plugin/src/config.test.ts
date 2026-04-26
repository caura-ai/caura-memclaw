import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  isMemorySlotClaimed,
  isMemclawFullyConfigured,
} from "./config.js";
import { getPluginDir } from "./paths.js";

// Minimal "happy-path" config scaffold — every predicate true. Individual
// tests selectively break one field at a time.
function happyConfig(): Record<string, unknown> {
  return {
    plugins: {
      allow: ["memclaw"],
      entries: { memclaw: { enabled: true } },
      load: { paths: [getPluginDir()] },
      slots: { memory: "memclaw" },
    },
    tools: { alsoAllow: [] },
  };
}

describe("isMemorySlotClaimed", () => {
  test("false when plugins.slots is missing", () => {
    const c = happyConfig();
    delete (c as any).plugins.slots;
    assert.equal(isMemorySlotClaimed(c), false);
  });

  test("false when memory slot is held by a different plugin", () => {
    const c = happyConfig();
    (c as any).plugins.slots.memory = "memory-core";
    assert.equal(isMemorySlotClaimed(c), false);
  });

  test("true when memory slot is memclaw", () => {
    assert.equal(isMemorySlotClaimed(happyConfig()), true);
  });
});

describe("isMemclawFullyConfigured", () => {
  // Paints the Fleet UI dashboard via heartbeat.setup_status.fully_configured.
  // Each test below corresponds to one of the four conditions that must hold.

  test("true on happy-path config", () => {
    assert.equal(isMemclawFullyConfigured(happyConfig()), true);
  });

  test("false when memclaw is not allowlisted", () => {
    const c = happyConfig();
    (c as any).plugins.allow = [];
    assert.equal(isMemclawFullyConfigured(c), false);
  });

  test("false when memclaw is disabled", () => {
    const c = happyConfig();
    (c as any).plugins.entries.memclaw.enabled = false;
    assert.equal(isMemclawFullyConfigured(c), false);
  });

  test("false when plugin path is not loaded", () => {
    const c = happyConfig();
    (c as any).plugins.load.paths = [];
    assert.equal(isMemclawFullyConfigured(c), false);
  });

  test("false when memory slot is not claimed", () => {
    const c = happyConfig();
    (c as any).plugins.slots.memory = "memory-core";
    assert.equal(isMemclawFullyConfigured(c), false);
  });
});
