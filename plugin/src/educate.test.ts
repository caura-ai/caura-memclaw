import { test, describe, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { tmpdir } from "os";
import { educateAgents, writeEducationFiles } from "./index.js";
import {
  buildToolsMd,
  buildAgentsMd,
  discoverAgentWorkspaces,
} from "./educate.js";
import { MEMCLAW_TOOLS } from "./tools.js";
import { MEMORY_TYPES, STATUSES } from "./tool-definitions.js";

// Resolve the shared SKILL.md that ships with the plugin. Tests run from
// `plugin/dist/educate.test.js`; the skill lives at
// `plugin/skills/memclaw/SKILL.md`.
const __dirname = dirname(fileURLToPath(import.meta.url));
const SHARED_SKILL_PATH = join(__dirname, "..", "skills", "memclaw", "SKILL.md");
function readSharedSkill(): string {
  return readFileSync(SHARED_SKILL_PATH, "utf-8");
}

function makeTmpBase(): string {
  return mkdtempSync(join(tmpdir(), "educate-test-"));
}

describe("educateAgents", () => {
  const dirs: string[] = [];
  function tmpBase(): string {
    const d = makeTmpBase();
    dirs.push(d);
    return d;
  }
  afterEach(() => {
    for (const d of dirs) {
      try { rmSync(d, { recursive: true, force: true }); } catch {}
    }
    dirs.length = 0;
  });

  test("writes to default workspace", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });

    const result = educateAgents("hello agents", undefined, base);

    assert.equal(result.verified, 1);
    assert.equal(result.count, 1);
    assert.deepEqual(result.educated, ["workspace"]);
    assert.equal(result.failed.length, 0);

    const content = readFileSync(join(base, "workspace", "HEARTBEAT.md"), "utf-8");
    assert.ok(content.includes("hello agents"));
  });

  test("writes to per-agent workspaces", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspaces", "agent-1"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent-2"), { recursive: true });

    const result = educateAgents("learn this", undefined, base);

    assert.equal(result.verified, 2);
    assert.equal(result.count, 2);
    assert.ok(result.educated.includes("agent-1"));
    assert.ok(result.educated.includes("agent-2"));

    for (const agent of ["agent-1", "agent-2"]) {
      const content = readFileSync(join(base, "workspaces", agent, "HEARTBEAT.md"), "utf-8");
      assert.ok(content.includes("learn this"));
    }
  });

  test("writes to both default and per-agent workspaces", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent-1"), { recursive: true });

    const result = educateAgents("broadcast", undefined, base);

    assert.equal(result.verified, 2);
    assert.ok(result.educated.includes("workspace"));
    assert.ok(result.educated.includes("agent-1"));
  });

  test("filters by agentIds", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspaces", "agent-1"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent-2"), { recursive: true });

    const result = educateAgents("selective", ["agent-2"], base);

    assert.equal(result.verified, 1);
    assert.deepEqual(result.educated, ["agent-2"]);
  });

  test("appends with separator to existing content", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });
    writeFileSync(join(base, "workspace", "HEARTBEAT.md"), "existing stuff\n", "utf-8");

    educateAgents("new prompt", undefined, base);

    const content = readFileSync(join(base, "workspace", "HEARTBEAT.md"), "utf-8");
    assert.ok(content.includes("existing stuff"));
    assert.ok(content.includes("---"));
    assert.ok(content.includes("new prompt"));
  });

  test("returns zero when no workspaces exist", () => {
    const base = tmpBase();
    // empty dir, no workspace or workspaces/

    const result = educateAgents("nobody home", undefined, base);

    assert.equal(result.verified, 0);
    assert.equal(result.count, 0);
    assert.equal(result.educated.length, 0);
  });
});

describe("discoverAgentWorkspaces", () => {
  const dirs: string[] = [];
  function tmpBase(): string {
    const d = makeTmpBase();
    dirs.push(d);
    return d;
  }
  afterEach(() => {
    for (const d of dirs) {
      try { rmSync(d, { recursive: true, force: true }); } catch {}
    }
    dirs.length = 0;
  });

  test("finds the default <baseDir>/workspace as id=main", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });

    const ws = discoverAgentWorkspaces(base);

    assert.equal(ws.length, 1);
    assert.equal(ws[0].id, "main");
    assert.equal(ws[0].path, join(base, "workspace"));
  });

  test("finds <baseDir>/workspace-<name> as id=<name> (hyphen-prefix)", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace-alpha"), { recursive: true });
    mkdirSync(join(base, "workspace-beta"), { recursive: true });

    const ws = discoverAgentWorkspaces(base);
    const ids = ws.map((w) => w.id).sort();

    assert.deepEqual(ids, ["alpha", "beta"]);
  });

  test("finds <baseDir>/workspaces/<name>/ subdirs (the openclaw agents add path)", () => {
    // Regression: the previous writeEducationFiles discovery walked baseDir
    // entries with `startsWith("workspace")` and did NOT recurse into
    // <baseDir>/workspaces/, so subdir agents were silently skipped.
    const base = tmpBase();
    mkdirSync(join(base, "workspaces", "agent-a"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent-b"), { recursive: true });

    const ws = discoverAgentWorkspaces(base);
    const ids = ws.map((w) => w.id).sort();

    assert.deepEqual(ids, ["agent-a", "agent-b"]);
    for (const w of ws) {
      assert.ok(w.path.includes("/workspaces/"), `subdir agent path should be under workspaces/: ${w.path}`);
    }
  });

  test("does NOT return the literal <baseDir>/workspaces parent dir as a workspace", () => {
    // Regression: the pre-fix code matched `"workspaces".startsWith("workspace")`
    // and treated the plural parent dir itself as a workspace, leading to
    // phantom education files at <baseDir>/workspaces/{TOOLS,AGENTS}.md.
    const base = tmpBase();
    mkdirSync(join(base, "workspaces"), { recursive: true });

    const ws = discoverAgentWorkspaces(base);

    for (const w of ws) {
      assert.notEqual(w.path, join(base, "workspaces"));
    }
  });

  test("honors agents.list[].workspace from openclaw.json", () => {
    const base = tmpBase();
    const customWs = join(base, "custom-ws-dir");
    mkdirSync(customWs, { recursive: true });
    writeFileSync(
      join(base, "openclaw.json"),
      JSON.stringify({
        agents: { list: [{ id: "custom-agent", workspace: customWs }] },
      }),
      "utf-8",
    );

    const ws = discoverAgentWorkspaces(base);

    assert.equal(ws.length, 1);
    assert.equal(ws[0].id, "custom-agent");
    assert.equal(ws[0].path, customWs);
  });

  test("dedups overlapping discoveries — agents.list + workspaces/<name>", () => {
    const base = tmpBase();
    const wsPath = join(base, "workspaces", "agent2");
    mkdirSync(wsPath, { recursive: true });
    writeFileSync(
      join(base, "openclaw.json"),
      JSON.stringify({
        agents: { list: [{ id: "agent2", workspace: wsPath }] },
      }),
      "utf-8",
    );

    const ws = discoverAgentWorkspaces(base);

    assert.equal(ws.length, 1, "expected dedup to a single entry");
    assert.equal(ws[0].id, "agent2");
  });

  test("filters by agentIds when set is non-empty", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent2"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent3"), { recursive: true });

    const onlyAgent2 = discoverAgentWorkspaces(base, ["agent2"]);
    const ids = onlyAgent2.map((w) => w.id).sort();

    assert.deepEqual(ids, ["agent2"]);
  });

  test("rejects paths outside baseDir", () => {
    const base = tmpBase();
    const outside = mkdtempSync(join(tmpdir(), "outside-"));
    dirs.push(outside);
    writeFileSync(
      join(base, "openclaw.json"),
      JSON.stringify({
        agents: { list: [{ id: "evil", workspace: outside }] },
      }),
      "utf-8",
    );

    const ws = discoverAgentWorkspaces(base);

    assert.equal(ws.length, 0, "path outside baseDir must be rejected");
  });
});

describe("writeEducationFiles", () => {
  // These are regression tests for the multi-agent education bug
  // (memclaw memory 90b7d579-068a-4561-a740-73a76a74b1ac). The old
  // implementation discovered workspaces via `readdirSync(baseDir)` filtered
  // by `startsWith("workspace")`, which silently skipped agents living under
  // `<baseDir>/workspaces/<name>/` and also wrote phantom education files
  // into `<baseDir>/workspaces/{TOOLS,AGENTS}.md` because the literal
  // `workspaces` directory matched the filter.

  const dirs: string[] = [];
  function tmpBase(): string {
    const d = makeTmpBase();
    dirs.push(d);
    return d;
  }
  afterEach(() => {
    for (const d of dirs) {
      try { rmSync(d, { recursive: true, force: true }); } catch {}
    }
    dirs.length = 0;
  });

  function setupAllFourPatterns(base: string): {
    main: string;
    hyphen: string;
    customFromConfig: string;
    subdir: string;
  } {
    const main = join(base, "workspace");
    const hyphen = join(base, "workspace-hyphenated");
    const customFromConfig = join(base, "agents", "custom-place");
    const subdir = join(base, "workspaces", "agent2");
    for (const d of [main, hyphen, customFromConfig, subdir]) {
      mkdirSync(d, { recursive: true });
    }
    writeFileSync(
      join(base, "openclaw.json"),
      JSON.stringify({
        agents: {
          list: [
            { id: "main" },
            { id: "configured", workspace: customFromConfig },
            { id: "agent2", workspace: subdir },
          ],
        },
      }),
      "utf-8",
    );
    return { main, hyphen, customFromConfig, subdir };
  }

  test("educates ALL four workspace patterns (the bug fix)", () => {
    const base = tmpBase();
    const ws = setupAllFourPatterns(base);

    const result = writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.equal(result.toolsUpdated, 4, `expected 4 TOOLS.md writes, got ${result.toolsUpdated}`);
    assert.equal(result.agentsUpdated, 4, `expected 4 AGENTS.md writes, got ${result.agentsUpdated}`);

    for (const wsPath of [ws.main, ws.hyphen, ws.customFromConfig, ws.subdir]) {
      const tools = readFileSync(join(wsPath, "TOOLS.md"), "utf-8");
      const agents = readFileSync(join(wsPath, "AGENTS.md"), "utf-8");
      assert.ok(tools.includes("MemClaw"), `TOOLS.md missing MemClaw section in ${wsPath}`);
      assert.ok(agents.includes("## Memory V2"), `AGENTS.md missing '## Memory V2' anchor in ${wsPath}`);
    }
  });

  test("subdir-of-plural agents are no longer silently skipped (regression)", () => {
    const base = tmpBase();
    const subdir = join(base, "workspaces", "agent-x");
    mkdirSync(subdir, { recursive: true });

    const result = writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.equal(result.toolsUpdated, 1);
    assert.equal(result.agentsUpdated, 1);
    assert.ok(readFileSync(join(subdir, "TOOLS.md"), "utf-8").includes("MemClaw"));
    assert.ok(readFileSync(join(subdir, "AGENTS.md"), "utf-8").includes("## Memory V2"));
  });

  test("does NOT create phantom files at <baseDir>/workspaces/{TOOLS,AGENTS}.md", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspaces", "agent-x"), { recursive: true });

    writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.ok(
      !existsSync(join(base, "workspaces", "TOOLS.md")),
      "phantom TOOLS.md must NOT be created at <baseDir>/workspaces/",
    );
    assert.ok(
      !existsSync(join(base, "workspaces", "AGENTS.md")),
      "phantom AGENTS.md must NOT be created at <baseDir>/workspaces/",
    );
  });

  test("idempotent: re-running does not double-append", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });

    writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);
    const firstTools = readFileSync(join(base, "workspace", "TOOLS.md"), "utf-8");
    const firstAgents = readFileSync(join(base, "workspace", "AGENTS.md"), "utf-8");

    const second = writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.equal(second.toolsUpdated, 0);
    assert.equal(second.agentsUpdated, 0);
    assert.equal(readFileSync(join(base, "workspace", "TOOLS.md"), "utf-8"), firstTools);
    assert.equal(readFileSync(join(base, "workspace", "AGENTS.md"), "utf-8"), firstAgents);
  });

  test("respects agentIds filter (educates only requested workspaces)", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspace"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent2"), { recursive: true });
    mkdirSync(join(base, "workspaces", "agent3"), { recursive: true });

    const result = writeEducationFiles(buildToolsMd(), buildAgentsMd(), ["agent2"], base);

    assert.equal(result.toolsUpdated, 1);
    assert.equal(result.agentsUpdated, 1);
    assert.ok(readFileSync(join(base, "workspaces", "agent2", "TOOLS.md"), "utf-8").includes("MemClaw"));
    assert.ok(!existsSync(join(base, "workspace", "TOOLS.md")) || !readFileSync(join(base, "workspace", "TOOLS.md"), "utf-8").includes("MemClaw"));
    assert.ok(!existsSync(join(base, "workspaces", "agent3", "TOOLS.md")) || !readFileSync(join(base, "workspaces", "agent3", "TOOLS.md"), "utf-8").includes("MemClaw"));
  });

  test("cleans up pre-existing phantom MemClaw files at <baseDir>/workspaces/", () => {
    const base = tmpBase();
    mkdirSync(join(base, "workspaces"), { recursive: true });
    writeFileSync(join(base, "workspaces", "TOOLS.md"), buildToolsMd(), "utf-8");
    writeFileSync(join(base, "workspaces", "AGENTS.md"), buildAgentsMd(), "utf-8");

    writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.ok(!existsSync(join(base, "workspaces", "TOOLS.md")));
    assert.ok(!existsSync(join(base, "workspaces", "AGENTS.md")));
  });

  test("cleanup leaves non-MemClaw content at <baseDir>/workspaces/ alone", () => {
    // Defensive: if a user happens to have unrelated TOOLS.md/AGENTS.md at
    // that path, we must not delete it. Cleanup is gated on memclaw markers.
    const base = tmpBase();
    mkdirSync(join(base, "workspaces"), { recursive: true });
    writeFileSync(join(base, "workspaces", "TOOLS.md"), "# my own tools notes\n", "utf-8");
    writeFileSync(join(base, "workspaces", "AGENTS.md"), "# my own agents notes\n", "utf-8");

    writeEducationFiles(buildToolsMd(), buildAgentsMd(), undefined, base);

    assert.equal(
      readFileSync(join(base, "workspaces", "TOOLS.md"), "utf-8"),
      "# my own tools notes\n",
    );
    assert.equal(
      readFileSync(join(base, "workspaces", "AGENTS.md"), "utf-8"),
      "# my own agents notes\n",
    );
  });
});

// ── Education file builder contracts ──
//
// These assertions encode role separation and transport-neutrality
// invariants. If any fail, the education content has drifted from the
// design captured in the comment block above the builders in educate.ts.

/** Host-specific words that must not appear in agent-facing education content. */
const FORBIDDEN_HOST_TERMS = ["plugin", "gateway", "openclaw"];

function assertTransportNeutral(content: string, label: string): void {
  const lower = content.toLowerCase();
  for (const term of FORBIDDEN_HOST_TERMS) {
    assert.ok(
      !lower.includes(term),
      `${label} leaked host-specific term "${term}"`,
    );
  }
}

describe("shared SKILL.md (plugin/skills/memclaw/SKILL.md)", () => {
  // SKILL.md is no longer generated by a builder. It ships as a static file
  // at the plugin root (`plugin/skills/memclaw/SKILL.md`), discovered by
  // OpenClaw via `openclaw.plugin.json:skills`. These tests pin the file's
  // presence and the invariants the content must hold.

  test("static file exists at the expected plugin-root path", () => {
    assert.ok(
      existsSync(SHARED_SKILL_PATH),
      `expected shared skill file at ${SHARED_SKILL_PATH}`,
    );
  });

  test("has OpenClaw-required frontmatter keys (name, description)", () => {
    const skill = readSharedSkill();
    assert.ok(skill.startsWith("---\n"), "missing YAML frontmatter delimiter");
    assert.ok(/\nname:\s*memclaw\b/.test(skill), "missing/wrong name: memclaw");
    assert.ok(/\ndescription:\s*\S/.test(skill), "missing description");
  });

  test("suppresses slash-command exposure (user-invocable: false)", () => {
    const skill = readSharedSkill();
    assert.ok(
      /\nuser-invocable:\s*false\b/.test(skill),
      "shared skill should set user-invocable: false to avoid a /memclaw slash command",
    );
  });

  test("gates on plugins.entries.memclaw.enabled via requires.config", () => {
    const skill = readSharedSkill();
    assert.ok(
      skill.includes("plugins.entries.memclaw.enabled"),
      "shared skill should be gated on plugin-enabled config path",
    );
    // metadata must be a single-line JSON object per OpenClaw's parser
    const metadataMatch = skill.match(/\nmetadata:\s*(\{.*\})\s*\n/);
    assert.ok(metadataMatch, "metadata frontmatter must be present and single-line");
    assert.doesNotThrow(
      () => JSON.parse(metadataMatch![1]),
      "metadata must be valid JSON",
    );
  });

  test("required body sections are present", () => {
    const skill = readSharedSkill();
    assert.ok(skill.includes("## Your identity"), "missing identity section");
    assert.ok(skill.includes("`agent_id`"), "missing agent_id");
    assert.ok(skill.includes("`fleet_id`"), "missing fleet_id");
    assert.ok(skill.includes("## The three rules"), "missing three rules");
    assert.ok(skill.includes("## Trust levels"), "missing trust section");
    assert.ok(skill.includes("## Sharing"), "missing sharing section");
    assert.ok(skill.includes("## Session loop"), "missing session loop");
  });

  test("identity section uses MUST language", () => {
    const skill = readSharedSkill();
    const idx = skill.indexOf("## Your identity");
    assert.ok(idx >= 0);
    const section = skill.slice(idx, skill.indexOf("## The three rules"));
    assert.ok(section.includes("MUST"), "identity section lacks MUST language");
  });

  test("Rule 3 describes delete as soft-delete requiring trust 3", () => {
    const skill = readSharedSkill();
    assert.ok(skill.includes("soft-delete"), "Rule 3 should say soft-delete");
    assert.ok(skill.includes("trust 3"), "Rule 3 should state trust level 3");
    assert.ok(
      !skill.toLowerCase().includes("hard delete") &&
        !skill.toLowerCase().includes("hard-delete"),
      "SKILL.md must not describe delete as hard-delete",
    );
  });

  test("holds the deep-dive tool reference relocated from TOOLS.md", () => {
    // The tool cards, decision tree, constraints, and error codes were moved
    // out of the bootstrap-every-turn TOOLS.md into this on-demand file as
    // part of the per-turn token-footprint reduction. Each section must land
    // here or the model loses the reference entirely.
    const skill = readSharedSkill();
    assert.ok(skill.includes("## Tool reference"), "missing ## Tool reference section");
    assert.ok(skill.includes("### Tool cards"), "missing ### Tool cards");
    assert.ok(skill.includes("### Which tool, when"), "missing ### Which tool, when");
    assert.ok(skill.includes("### Constraints that matter"), "missing ### Constraints that matter");
    assert.ok(skill.includes("### Error codes"), "missing ### Error codes");
  });

  test("all 9 tool cards are present in SKILL.md", () => {
    const skill = readSharedSkill();
    for (const tool of [
      "memclaw_recall", "memclaw_write", "memclaw_manage", "memclaw_list",
      "memclaw_doc", "memclaw_entity_get", "memclaw_tune",
      "memclaw_insights", "memclaw_evolve",
    ]) {
      assert.ok(
        skill.includes(`**\`${tool}(`) || skill.includes(`\`${tool}(`),
        `SKILL.md missing tool card for ${tool}`,
      );
    }
  });

  test("error codes appear verbatim in SKILL.md", () => {
    const skill = readSharedSkill();
    for (const code of ["INVALID_ARGUMENTS", "BATCH_TOO_LARGE", "INVALID_BATCH_ITEM"]) {
      assert.ok(skill.includes(code), `SKILL.md missing error code ${code}`);
    }
  });

  // Note: the shared SKILL.md is intentionally a plugin artifact — it is
  // gated on `plugins.entries.memclaw.enabled` and its footer references
  // the plugin install path. The transport-neutrality check does NOT apply
  // here (it still applies to TOOLS.md / AGENTS.md, which append to
  // workspace-owned files that are loaded across transports).
});

describe("buildToolsMd", () => {
  const ALL_TOOLS = [
    "memclaw_recall",
    "memclaw_write",
    "memclaw_manage",
    "memclaw_list",
    "memclaw_doc",
    "memclaw_entity_get",
    "memclaw_tune",
    "memclaw_insights",
    "memclaw_evolve",
  ];

  test("lists all 9 tools", () => {
    const tools = buildToolsMd();
    for (const tool of ALL_TOOLS) {
      assert.ok(tools.includes(tool), `missing tool: ${tool}`);
    }
  });

  test("contains exactly the lean sections (Quick Matrix + Vocabulary)", () => {
    // TOOLS.md is intentionally lean: it ships only the quick matrix (so the
    // model knows what tools exist each turn) and the vocabulary table (so
    // sub-agents — which receive only AGENTS.md and TOOLS.md — still have
    // enum values without needing to read SKILL.md). Everything else (tool
    // cards, decision tree, constraints, error codes) lives in SKILL.md.
    const tools = buildToolsMd();
    assert.ok(tools.includes("### Quick matrix"), "missing quick matrix");
    assert.ok(tools.includes("### Vocabulary"), "missing vocabulary");
    // Deep-dive sections must NOT appear in TOOLS.md — they were relocated
    // to SKILL.md as part of the per-turn token-footprint reduction.
    assert.ok(!tools.includes("### Tool cards"), "tool cards must live in SKILL.md, not TOOLS.md");
    assert.ok(!tools.includes("### Which tool, when"), "decision tree must live in SKILL.md, not TOOLS.md");
    assert.ok(!tools.includes("### Constraints"), "constraints must live in SKILL.md, not TOOLS.md");
    assert.ok(!tools.includes("### Error codes"), "error codes must live in SKILL.md, not TOOLS.md");
  });

  test("points readers to SKILL.md for the deep dive", () => {
    const tools = buildToolsMd();
    assert.ok(
      tools.includes("skills/memclaw/SKILL.md"),
      "TOOLS.md must point readers at SKILL.md for per-tool signatures and error codes",
    );
  });

  test("vocabulary covers every memory_type from the SoT (drift guard)", () => {
    // If this fails, either MEMORY_TYPES in tool-definitions.ts changed (update
    // TOOLS.md), or someone trimmed the TOOLS.md vocabulary row (don't — the
    // model needs the full enumeration).
    const tools = buildToolsMd();
    for (const t of MEMORY_TYPES) {
      assert.ok(tools.includes(t), `vocabulary missing memory_type "${t}"`);
    }
  });

  test("vocabulary covers every status from the SoT (drift guard)", () => {
    const tools = buildToolsMd();
    for (const s of STATUSES) {
      assert.ok(tools.includes(s), `vocabulary missing status "${s}"`);
    }
  });

  test("vocabulary covers the remaining schema-only enums", () => {
    const tools = buildToolsMd();
    for (const v of ["scope_agent", "scope_team", "scope_org"]) {
      assert.ok(tools.includes(v), `vocabulary missing visibility "${v}"`);
    }
    for (const m of ["fast", "strong"]) {
      assert.ok(tools.includes(m), `vocabulary missing write_mode "${m}"`);
    }
    for (const f of ["contradictions", "divergence", "discover"]) {
      assert.ok(tools.includes(f), `vocabulary missing focus "${f}"`);
    }
  });

  test("vocabulary has a fleet_ids row (recall cross-fleet filter)", () => {
    const tools = buildToolsMd();
    assert.ok(
      tools.includes("`fleet_ids`"),
      "vocabulary missing fleet_ids entry",
    );
  });

  test("quick-matrix header uses dynamic tool count", () => {
    const tools = buildToolsMd();
    assert.ok(
      tools.includes(`### Quick matrix · ${MEMCLAW_TOOLS.length} tools`),
      `quick matrix header missing dynamic tool count (expected "### Quick matrix · ${MEMCLAW_TOOLS.length} tools")`,
    );
  });

  test("is transport-neutral", () => {
    assertTransportNeutral(buildToolsMd(), "TOOLS.md");
  });
});

describe("buildAgentsMd", () => {
  test("contains the '## Memory V2' idempotency anchor", () => {
    // writeEducationFiles() skips append when the workspace AGENTS.md
    // already contains "## Memory V2". Breaking this anchor would cause
    // duplicate appends on re-education.
    const agents = buildAgentsMd();
    assert.ok(
      agents.includes("## Memory V2"),
      "AGENTS.md section lost the '## Memory V2' anchor; writeEducationFiles idempotency will break",
    );
  });

  test("contains identity mandate as a first-class section", () => {
    const agents = buildAgentsMd();
    assert.ok(agents.includes("### Identity is non-negotiable"));
    assert.ok(agents.includes("`agent_id`"));
    assert.ok(agents.includes("`fleet_id`"));
    assert.ok(agents.includes("MUST"));
  });

  test("write triggers are a structured list, not a comma string", () => {
    const agents = buildAgentsMd();
    assert.ok(agents.includes("### Write triggers"));
    // All canonical triggers present as bullet lines.
    for (const trigger of [
      "Task completed",
      "Bug found or fixed",
      "Deployment performed",
      "Decision made",
      "API discovered or changed",
      "Person context updated",
      "Blocker encountered",
      "Commitment made",
      "Configuration changed",
      "Error pattern identified",
    ]) {
      assert.ok(agents.includes(trigger), `missing write trigger: ${trigger}`);
    }
  });

  test("contains the 3-layer capture, subagent protocol, and prohibited list", () => {
    const agents = buildAgentsMd();
    assert.ok(agents.includes("### 3-layer memory capture"));
    assert.ok(agents.includes("### Orchestrator + subagent protocol"));
    assert.ok(agents.includes("### Prohibited behaviors"));
    assert.ok(agents.includes("NEVER fabricate"));
    assert.ok(agents.includes("NEVER silently drop"));
  });

  test("delete prohibition uses soft-delete terminology (not hard-delete)", () => {
    // Per review: `op=delete` is a soft-delete, not a hard-delete. The
    // prohibition should say so, and the phrase "hard-delete" must not
    // appear anywhere in AGENTS.md.
    const agents = buildAgentsMd();
    assert.ok(
      agents.includes("NEVER delete memories you merely disagree with"),
      "missing the 'merely disagree' framing for delete prohibition",
    );
    assert.ok(
      agents.includes("soft-delete"),
      "prohibition should clarify op=delete is a soft-delete",
    );
    assert.ok(
      !agents.toLowerCase().includes("hard-delete"),
      "AGENTS.md must not call op=delete a hard-delete",
    );
  });

  test("nudges the model to read SKILL.md before the first MemClaw call", () => {
    // The tool-reference deep dive lives in SKILL.md (loaded via `read` on
    // demand). AGENTS.md — injected as bootstrap every turn — must direct
    // the model to load SKILL.md before its first MemClaw tool call in a
    // session so the signatures, decision guidance, and error codes are
    // in context when needed.
    const agents = buildAgentsMd();
    assert.ok(
      agents.includes("skills/memclaw/SKILL.md"),
      "AGENTS.md must reference skills/memclaw/SKILL.md by path",
    );
    assert.ok(
      /before your first MemClaw/i.test(agents),
      "AGENTS.md must carry a 'before your first MemClaw call' nudge",
    );
  });

  test("is transport-neutral", () => {
    assertTransportNeutral(buildAgentsMd(), "AGENTS.md");
  });
});

describe("openclaw.plugin.json manifest", () => {
  // OpenClaw's plugin-skill resolver discovers SKILL.md via the `skills`
  // field in the manifest. If this field is dropped or the path is wrong,
  // the shared skill is invisible to agents even though the file exists.

  const MANIFEST_PATH = join(__dirname, "..", "openclaw.plugin.json");

  test("declares skills: [\"skills\"] so OpenClaw's resolver finds the shared SKILL.md", () => {
    assert.ok(existsSync(MANIFEST_PATH), `manifest not found at ${MANIFEST_PATH}`);
    const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf-8"));
    assert.ok(Array.isArray(manifest.skills), "manifest missing `skills` array");
    assert.ok(
      manifest.skills.includes("skills"),
      `manifest.skills should contain "skills" (got ${JSON.stringify(manifest.skills)})`,
    );
  });
});
