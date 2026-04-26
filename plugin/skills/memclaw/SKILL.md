---
name: memclaw
description: Persistent, cross-session, multi-agent memory. Semantic recall of decisions and findings; write outcomes; supersede facts when they change.
user-invocable: false
metadata: {"openclaw": {"requires": {"config": ["plugins.entries.memclaw.enabled"]}}}
---

# MemClaw Skill

Long-term memory that persists across sessions and is shared under access
controls. The primary place where decisions, findings, outcomes, and
learned rules live.

## Your identity: `agent_id` and `fleet_id`

You MUST identify correctly on every MemClaw call.

- `agent_id` — who you are. Attributes memories, drives trust progression,
  gates `scope_agent` privacy. Resolved by your runtime. Never fabricate,
  hardcode a placeholder, or impersonate another agent.
- `fleet_id` — your team / organization scope. Required for
  `visibility=scope_team`, for fleet-scoped reads, and for cross-fleet
  operations. Never substitute another team's `fleet_id`.

Wrong `agent_id` poisons attribution and trust. Wrong `fleet_id` leaks
memories to the wrong team or hides them from your own. If either is
uncertain, do NOT guess — read from the runtime, ask the orchestrator,
or write privately (`visibility=scope_agent`) until resolved.

## The three rules

**Rule 1 — Recall before you start.** Never start cold. Begin every
meaningful task with a semantic recall: "what is already known about
this?"

**Rule 2 — Write when something matters.** After completing work, or when
something important happens mid-task, write a memory. Supply raw prose —
the server auto-classifies type, summary, tags, dates. Include names,
paths, numbers, outcomes. Skip vague observations and intermediate steps.
Checkpoint every 30 minutes on long tasks. Batch multiple discrete
records into one call.

**Rule 3 — Supersede, don't delete.** For a changed fact: (1) write the
new one, (2) recall the old one, (3) transition the old to `outdated`.
Reserve deletes (soft-delete, requires trust 3) for correcting genuinely
wrong data.

## Trust levels

Auto-registered at trust 1 on your first write.

| Level | Name        | Read                      | Write                  |
|:-----:|-------------|---------------------------|------------------------|
| 0     | restricted  | —                         | —                      |
| 1     | standard    | own fleet                 | own fleet              |
| 2     | cross-fleet | all fleets in your tenant | own fleet              |
| 3     | admin       | all                       | all, including deletes |

Scope-based escalation:
- browsing or reflecting with `scope="fleet"` / `"all"` → trust 2
- reporting outcomes (`memclaw_evolve`) → trust 2
- `memclaw_manage op=delete` → trust 3

If denied, surface the error; do not silently retry with a narrower scope.

## Sharing: visibility and scope

**Visibility (on write):** `scope_agent` (private) · `scope_team`
*(default — your fleet)* · `scope_org` (all fleets in tenant).

**Scope (on read — `_list` / `_insights`):** `agent` *(default)* ·
`fleet` (trust 2) · `all` (trust 2).

Prefer `scope_team` when writing; prefer `scope=agent` when reading
unless you need cross-agent context.

## Containers

- **Memory** — unstructured, findable semantically. Decisions, observations, rules, outcomes, recaps.
- **Doc** — structured record with a natural key (`collection + doc_id`). Customers, configs, task lists, inventories.
- **Entity** — named graph object (person, project, service). Fetch by UUID from a prior recall.

If you need semantic search, it's a memory. If you need keyed lookup,
it's a doc. If you already hold an ID, it's an entity.

## Good memories

Dated, concrete, standalone, atomic, updated (not duplicated).

## Session loop

1. Recall — "what is known about this?" / "what happened since last session?"
2. Work — act on the recalled context.
3. Write — at checkpoints and session end.
4. Evolve — if you acted on specific memories, report the outcome (trust 2).

---

## Tool reference

The at-a-glance tool list and enum vocabulary live in `TOOLS.md` in your
workspace (bootstrap-injected every turn). This section holds the per-tool
signatures, decision guidance, constraints, and error codes — load it before
your first MemClaw call in a session.

### Tool cards

**`memclaw_recall(query, top_k=5, include_brief=false, memory_type=?, status=?, filter_agent_id=?, fleet_ids=?)`**
Hybrid semantic+keyword search. For metadata browse → `memclaw_list`;
for a known id → `memclaw_manage(op="read")`. `include_brief=true` adds
an LLM-summarized paragraph.

**`memclaw_write(content=? | items=?, visibility="scope_team", memory_type=?, weight=?, metadata=?, write_mode="auto", source_uri=?, run_id=?)`**
Provide exactly one of `content` / `items`. Server auto-classifies.
`items` batches up to 100. `write_mode`: `fast` skips embed
(keyword-only recall later); `strong` forces LLM enrichment; `auto` is
usually right.

**`memclaw_manage(op, memory_id, ...)`** — op ∈ {read, update, transition, delete}
- `update`: patch `content` / `memory_type` / `weight` / `title` / `metadata` / `source_uri`; re-embeds if content changes.
- `transition`: set `status` (see Vocabulary in `TOOLS.md`).
- `delete`: soft-delete; trust 3. Prefer `transition` to `outdated` / `archived`.

**`memclaw_list(scope="agent", memory_type=?, written_by=?, status=?, weight_min/max=?, created_after/before=?, sort="created_at", order="desc", limit=25, cursor=?)`**
Non-semantic enumeration. Cursor pagination requires
`sort=created_at order=desc`. `scope="fleet"` / `"all"` → trust 2.

**`memclaw_doc(op, collection, doc_id=?, data=?, where=?, order_by=?, limit=20, offset=0)`** — op ∈ {write, read, query, delete}
Structured records by `collection + doc_id`. `write` upserts.

**`memclaw_entity_get(entity_id)`**
UUID from a prior call — never fabricate.

**`memclaw_tune(top_k?, min_similarity?, fts_weight?, freshness_floor?, freshness_decay_days?, recall_boost_cap?, recall_decay_window_days?, graph_max_hops?, similarity_blend?)`**
Persists — affects every future `memclaw_recall`. No fields → read the
current profile. Change one or two at a time. `fts_weight` 0 = pure
semantic, 1 = pure keyword.

**`memclaw_insights(focus, scope="agent", fleet_id=?)`**
Reflection. `focus` ∈ {contradictions, failures, stale, divergence,
patterns, discover}. Results saved as `insight` memories. Run at
boundaries, not every turn. `scope="fleet"` / `"all"` → trust 2;
`focus="divergence"` requires non-agent scope.

**`memclaw_evolve(outcome, outcome_type, related_ids=?)`**
Close the loop. `outcome_type` ∈ {success, failure, partial}.
`related_ids` = the recall IDs you acted on. Success reinforces weights;
failure auto-creates `rule` memories. Trust 2.

### Which tool, when

- Might have seen before → `memclaw_recall`
- Enumerate by filter / date / author → `memclaw_list`
- Already hold the ID → `memclaw_manage op=read` or `memclaw_entity_get`
- Record a fact / decision / event / outcome → `memclaw_write`
- Structured record with a key → `memclaw_doc`
- Fact no longer true → `memclaw_write` (new) + `memclaw_manage op=transition status=outdated` (old)
- Acted on a recalled memory → `memclaw_evolve`
- Recall quality off across queries → `memclaw_tune` (once, sticky)
- Session boundary / orchestrator sweep → `memclaw_insights`

### Constraints that matter

- `memclaw_write`: exactly one of `content` / `items`; never both.
- `items` capped at 100 → `BATCH_TOO_LARGE`.
- Supersede via `transition`; reserve `delete` (trust 3) for wrong data.
- Cursor pagination needs `sort=created_at` + `order=desc`.
- `_entity_get` / `_manage` use real UUIDs — never invent.
- `_tune` persists; do not call per-query.

### Error codes

`INVALID_ARGUMENTS` · `BATCH_TOO_LARGE` · `INVALID_BATCH_ITEM`. Other
errors surface with HTTP status + message — return them to your caller,
do not swallow.

---

*This skill ships with the MemClaw plugin at its install path; it is visible
to every agent on the node that has the plugin enabled. To customize it for
a specific agent, place a replacement file at
`<workspace>/skills/memclaw/SKILL.md` in that agent's workspace — it takes
precedence over this shared copy.*
