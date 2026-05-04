/**
 * Tests for the MemClaw plugin tool surface.
 *
 * Guards the contract between `plugin/tools.json` (SoT), `MEMCLAW_TOOLS`
 * (registration order), `PARAM_SCHEMAS` (runtime input validation), and
 * `ENDPOINT_DISPATCH` (HTTP routing).
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { MEMCLAW_TOOLS } from "./tools.js";
import { createToolFromSpec } from "./tool-definitions.js";
import { TOOL_SPECS, TOOL_SPECS_BY_NAME, getSpec } from "./tool-specs.js";

describe("tool-specs loader", () => {
  test("loads a non-empty ordered spec list from tools.json", () => {
    assert.ok(Array.isArray(TOOL_SPECS));
    assert.ok(TOOL_SPECS.length > 0);
    for (const spec of TOOL_SPECS) {
      assert.equal(typeof spec.name, "string");
      assert.ok(spec.name.startsWith("memclaw_"), `spec name ${spec.name}`);
      assert.equal(typeof spec.description, "string");
      assert.ok(spec.description.length > 0, `${spec.name}: empty description`);
      assert.equal(typeof spec.plugin_exposed, "boolean");
    }
  });

  test("name index matches TOOL_SPECS exactly", () => {
    const fromIndex = new Set(Object.keys(TOOL_SPECS_BY_NAME));
    const fromList = new Set(TOOL_SPECS.map((s) => s.name));
    assert.deepEqual(fromIndex, fromList);
  });

  test("getSpec throws for unknown tool", () => {
    assert.throws(() => getSpec("memclaw_not_a_thing"), /Unknown tool/);
  });

  test("getSpec returns matching entry for known tool", () => {
    const spec = getSpec("memclaw_recall");
    assert.equal(spec.name, "memclaw_recall");
    assert.equal(spec.plugin_exposed, true);
  });
});

describe("MEMCLAW_TOOLS surface", () => {
  test("is the expected list of plugin tools", () => {
    assert.deepEqual([...MEMCLAW_TOOLS], [
      "memclaw_recall",
      "memclaw_write",
      "memclaw_manage",
      "memclaw_doc",
      "memclaw_list",
      "memclaw_entity_get",
      "memclaw_tune",
      "memclaw_insights",
      "memclaw_evolve",
      "memclaw_stats",
    ]);
  });

  test("every listed tool is plugin_exposed in tools.json", () => {
    for (const name of MEMCLAW_TOOLS) {
      const spec = TOOL_SPECS_BY_NAME[name];
      assert.ok(spec, `${name} missing from tools.json`);
      assert.equal(spec.plugin_exposed, true);
    }
  });

  test("every plugin_exposed tool in tools.json is listed in MEMCLAW_TOOLS", () => {
    const exposed = TOOL_SPECS.filter((s) => s.plugin_exposed).map((s) => s.name);
    for (const name of exposed) {
      assert.ok(
        (MEMCLAW_TOOLS as readonly string[]).includes(name),
        `${name} is plugin_exposed in tools.json but absent from MEMCLAW_TOOLS`,
      );
    }
  });

  test("STM and placeholder tools are NOT in MEMCLAW_TOOLS", () => {
    for (const hidden of [
      "memclaw_notes_read",
      "memclaw_bulletin_read",
      "memclaw_promote",
    ]) {
      assert.ok(
        !(MEMCLAW_TOOLS as readonly string[]).includes(hidden),
        `${hidden} should not be plugin-exposed`,
      );
    }
  });
});

describe("createToolFromSpec factory", () => {
  test("produces a valid AgentTool for every listed name", () => {
    for (const name of MEMCLAW_TOOLS) {
      const tool = createToolFromSpec(name);
      assert.equal(tool.name, name);
      assert.ok(tool.label.startsWith("MemClaw "), `${name}: label`);
      assert.ok(tool.description.length > 0, `${name}: description`);
      assert.ok(
        typeof tool.parameters === "object" && tool.parameters !== null,
        `${name}: parameters is object`,
      );
      assert.equal(typeof tool.execute, "function");
    }
  });

  test("throws for a tool name not in tools.json", () => {
    assert.throws(
      () => createToolFromSpec("memclaw_does_not_exist"),
      /Unknown tool/,
    );
  });

  test("op-dispatched tools declare op + required path params", () => {
    const manage = createToolFromSpec("memclaw_manage").parameters as any;
    assert.deepEqual(manage.required, ["op", "memory_id"]);
    assert.deepEqual(manage.properties.op.enum, [
      "read", "update", "transition", "delete",
    ]);

    const doc = createToolFromSpec("memclaw_doc").parameters as any;
    assert.deepEqual(doc.required, ["op", "collection"]);
    assert.deepEqual(doc.properties.op.enum, [
      "write", "read", "query", "delete",
    ]);
  });

  test("memclaw_write requires only agent_id (content/items are mutually exclusive)", () => {
    const write = createToolFromSpec("memclaw_write").parameters as any;
    assert.deepEqual(write.required, ["agent_id"]);
    assert.ok(write.properties.content);
    assert.ok(write.properties.items);
    assert.equal(write.properties.items.maxItems, 100);
  });

  test("memclaw_list has no required params (trust gate handled server-side)", () => {
    const list = createToolFromSpec("memclaw_list").parameters as any;
    assert.deepEqual(list.required, []);
    assert.ok(list.properties.cursor);
    assert.ok(list.properties.include_deleted);
  });

  test("description falls back to tools.json value when no live override", () => {
    // `getToolDescription` reads from the shared cache in env.ts. On a
    // fresh import (no /tool-descriptions fetch yet), the fallback should
    // be the description baked into tools.json.
    const spec = getSpec("memclaw_recall");
    const tool = createToolFromSpec("memclaw_recall");
    assert.equal(tool.description, spec.description);
  });
});

describe("labelFor naming conversion", () => {
  test("memclaw_doc → MemClaw Doc, memclaw_entity_get → MemClaw Entity Get", () => {
    assert.equal(createToolFromSpec("memclaw_doc").label, "MemClaw Doc");
    assert.equal(
      createToolFromSpec("memclaw_entity_get").label,
      "MemClaw Entity Get",
    );
    assert.equal(createToolFromSpec("memclaw_list").label, "MemClaw List");
    assert.equal(
      createToolFromSpec("memclaw_manage").label,
      "MemClaw Manage",
    );
  });
});
