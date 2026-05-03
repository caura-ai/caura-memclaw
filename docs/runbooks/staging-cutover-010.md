# Staging cutover — alembic 010 (vector_dim 768 → 1024)

**Owner:** Platform Eng (TBD)  **Last reviewed:** 2026-05-03  **Estimated wall-clock:** 1–2 h (plus 7-day soak)

> Read all of this before starting. Each `##` section ends with a verification
> check — DO NOT proceed to the next section without a green result.
>
> This runbook migrates **staging** from the 768-dim OpenAI baseline to the
> 1024-dim `BAAI/bge-m3` (or hosted-OpenAI-with-`dimensions=1024`) configuration.
> Migration 010 is **destructive**: every existing `embedding` value on
> `memories`, `entities.name_embedding`, and `documents.embedding` is set to
> NULL before the column is widened. Search returns nothing for affected rows
> until the backfill (section 4) re-embeds them.
>
> Production cutover is a separate runbook (Spec F). Do not run this against
> prod.

## Pre-flight checklist

- [ ] PR #51 (`CAURA-444-local-embed-infra`) is merged to `caura-memclaw:main` (includes Spec A safety gate and Spec B release notes).
- [ ] Spec C's backfill task PR is merged to `caura-memclaw:main` and the `core-worker` image with `python -m core_worker.cli backfill-embeddings` has been built.
- [ ] Spec D's pre-flight script PR is merged to `caura-memclaw:main` (`core_storage_api.scripts.preflight_010`).
- [ ] Enterprise-side PR adding the `destructive` input to `migrate-db.yml` is merged on `caura-memclaw-enterprise:dev`.
- [ ] Staging AlloyDB has a verified backup taken in the last 24 h. Cloud SQL automated backups are normally enough; confirm the most-recent point-in-time recovery target in the GCP console.
- [ ] You have `gcloud` auth on `alpine-theory-469016-c8` and `gh` auth on `caura-ai/caura-memclaw-enterprise`.
- [ ] You have a Slack channel ready for status updates and a 30-min hold on staging traffic from QA / connected demo apps.
- [ ] Staging Cloud Run services are healthy *before* you start: `staging-memclaw-core-api`, `staging-memclaw-core-storage`, and (if used) `staging-memclaw-platform-storage`.

## 1. Run pre-flight against staging DB

Read-only. Reports row counts that would be NULL'd, estimated wall-clock for
the destructive UPDATEs, and the exact opt-in workflow_dispatch invocation.

```bash
# From a workstation with gcloud auth and access to the staging VPC
# (use the Cloud SQL Auth Proxy or a bastion if direct connection is blocked).

gcloud sql connect alpine-theory-469016-c8-staging-pg --user=memclaw --quiet \
  -- -t -c "\\!python -m core_storage_api.scripts.preflight_010 --dsn $DATABASE_URL"
```

Capture the output. **Stop here and read it carefully:**

- "Rows to NULL" total: should be a number you recognize as roughly the count of memories in staging.
- "Est. UPDATE": rough ETA. If it's >10 minutes, plan accordingly (announce a quiet window in Slack).
- "Alembic head": should read `009`. If it already says `010`, the migration has been applied; skip section 2 and re-run section 4 (backfill) only.

## 2. Trigger migration via Cloud Run Job

Manual dispatch of `migrate-db.yml` with `destructive=true`. The `destructive`
input is propagated to the OSS migration job as
`MEMCLAW_RUN_DESTRUCTIVE_MIGRATIONS=true`, satisfying the Spec A safety gate
in migration 010.

```bash
gh workflow run migrate-db.yml \
  --repo caura-ai/caura-memclaw-enterprise \
  --ref dev \
  -f target_revision=010 \
  -f destructive=true
```

Watch progress:

```bash
gh run watch --repo caura-ai/caura-memclaw-enterprise
```

**Verification:**

```bash
# After job completes, the alembic_version table on staging should show 010.
gcloud sql connect alpine-theory-469016-c8-staging-pg --user=memclaw --quiet \
  -- -c "SELECT version_num FROM alembic_version;"
```

Expect: `010`.

**Failure modes:**

- `RuntimeError: alembic 010_vector_dim_1024 is destructive...` — the
  `destructive=true` input wasn't propagated. Check the workflow YAML diff and
  confirm the `--set-env-vars` line in the OSS migration step includes
  `MEMCLAW_RUN_DESTRUCTIVE_MIGRATIONS=...`.
- `lock_timeout` during `UPDATE ... SET embedding = NULL` — retry the workflow.
  The migration is idempotent: rows already NULL'd are skipped by the
  `WHERE embedding IS NOT NULL` predicate, and the `autocommit_block` in 010
  resumes cleanly.
- `ALTER TYPE` fails with `expected 1024 dimensions, not 768` — a row was
  written with a 768-dim embedding between the UPDATE and the ALTER (race
  with live traffic). Re-run the workflow; the next attempt re-NULLs the
  straggler.

## 3. Roll out the new image (core-api + core-storage-api) to staging Cloud Run

Push the latest `main` build (post-PR-#51) to staging. The
`deploy-oss-services.yml` workflow runs on push to `dev` and tags images with
the commit SHA, then deploys to `staging-memclaw-core-api` and
`staging-memclaw-core-storage`.

```bash
# In caura-memclaw-enterprise:dev. Bump the OSS pin to the merged PR-#51 sha.
# (Implementation depends on how OSS images are pinned in the enterprise repo
# — likely a tag/SHA in deploy-oss-services.yml or a manifest. Fill in for the
# team's actual mechanism, then push to dev to trigger the deploy.)
```

**Verification:**

- `gh run watch --repo caura-ai/caura-memclaw-enterprise` shows `deploy-oss-services` green.
- Both services rolled forward:
  ```bash
  for svc in staging-memclaw-core-api staging-memclaw-core-storage; do
    echo "$svc:"
    gcloud run services describe "$svc" --region=us-central1 \
      --format='value(status.latestReadyRevisionName,spec.template.spec.containers[0].image)'
  done
  ```
- Health endpoint OK and config reflects `VECTOR_DIM=1024`:
  ```bash
  CORE_API_URL=$(gcloud run services describe staging-memclaw-core-api \
    --region=us-central1 --format='value(status.url)')
  curl -fsS "$CORE_API_URL/api/v1/health"
  ```
  Expect `{"status":"ok",...}`.

## 4. Run the embedding backfill

The backfill is a one-shot Cloud Run Job invoking the same code shipped in
the `core-worker` image. It scans `WHERE embedding IS NULL` and publishes
`EMBED_REQUESTED` events for each row. Idempotent and restartable.

```bash
RUN_ID=$(date +%s)
CORE_WORKER_IMAGE=$(gcloud run services describe staging-memclaw-core-storage \
  --region=us-central1 --format='value(spec.template.spec.containers[0].image)' \
  | sed 's|/core-storage|/core-worker|')   # adjust if image-naming differs

gcloud run jobs create memclaw-backfill-${RUN_ID} \
  --image="${CORE_WORKER_IMAGE}" \
  --region=us-central1 \
  --vpc-connector=memclaw-staging-vpc \
  --vpc-egress=private-ranges-only \
  --command=python \
  --args=-m,core_worker.cli,backfill-embeddings,--batch-size=500,--max-inflight=100 \
  --set-env-vars="..."   # mirror the envs of the running core-worker service \
  --task-timeout=3600s \
  --max-retries=0

gcloud run jobs execute memclaw-backfill-${RUN_ID} --region=us-central1 --wait
```

**Verification (mid-run):** tail the job logs and watch the
`embedding_backfill progress: scanned=... published=...` lines. The
`published` count should monotonically increase. Final report:
`scanned=N published=N elapsed=...s`.

```bash
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=memclaw-backfill-${RUN_ID}" \
  --limit=50 --format='value(textPayload)' --freshness=10m
```

**Verification (post-run):** sample the DB to confirm coverage:

```bash
gcloud sql connect alpine-theory-469016-c8-staging-pg --user=memclaw --quiet \
  -- -c "SELECT COUNT(*) AS total,
                COUNT(embedding) AS with_embedding,
                COUNT(*) - COUNT(embedding) AS missing
         FROM memories;"
```

Expect: `missing` is 0 (or only newly-written-but-not-yet-embedded rows from
live traffic). Repeat for `entities.name_embedding` and `documents.embedding`.

If `published < scanned`, some rows failed to publish — check `core-worker`
logs for the offending IDs and re-run the backfill (idempotent — it will only
re-publish the still-NULL rows).

## 5. Validate retrieval quality

Run the LongMemEval bench against staging — the same harness used in PR #51's
bench-off. The acceptance bar is set from the PR-#51 N=30 result on
`single-session-user` with `BAAI/bge-m3`: R@5 = 0.963, accuracy = 93.3 %.

```bash
cd /Users/rantaig/caura/local_emb_res
python runner_memclaw_lme.py \
  -t single-session-user -d 0 -n 30 --seed 42 --top-k 5 --settle-time 5 \
  --concurrency 1 \
  --url "$(gcloud run services describe staging-memclaw-core-api \
            --region=us-central1 --format='value(status.url)')" \
  --api-key <staging-admin-key> \
  --data-path ./longmemeval_s_cleaned.json \
  -o ./results/staging-bge-m3-cutover/
```

**Verification:**

- R@5 ≥ 0.95 — within ~1 pp of the PR-#51 baseline.
- accuracy ≥ 0.90.

If either drops below: **STOP**, escalate in the Slack channel, and consider
rollback per `docs/runbooks/rollback-010.md` (Spec G).

## 6. Cutover the embedder env (only if switching to TEI)

If this cutover includes pointing core-api at the self-hosted TEI sidecar
(Phase-2 of `local_emb_res/design_integration.md`), update the four envs on
the core-api Cloud Run service:

```bash
gcloud run services update staging-memclaw-core-api --region=us-central1 \
  --update-env-vars=OPENAI_EMBEDDING_BASE_URL=http://tei.staging:80/v1 \
  --update-env-vars=OPENAI_EMBEDDING_MODEL=BAAI/bge-m3 \
  --update-env-vars=OPENAI_EMBEDDING_SEND_DIMENSIONS=false
```

If staging is staying on hosted OpenAI with `dimensions=1024`, **skip this
step** — the migration in section 2 plus the image rollout in section 3 is
the cutover.

**Verification:** smoke-test write → search round-trip:

```bash
CORE_API_URL=$(gcloud run services describe staging-memclaw-core-api \
  --region=us-central1 --format='value(status.url)')

curl -fsS -X POST "$CORE_API_URL/api/v1/memories" \
  -H 'X-API-Key: <key>' -H 'content-type: application/json' \
  -d '{"tenant_id":"smoke-cutover","agent_id":"runbook","content":"cutover smoke check"}'

# Wait ~5s for the worker to embed.
sleep 5

curl -fsS -X POST "$CORE_API_URL/api/v1/search" \
  -H 'X-API-Key: <key>' -H 'content-type: application/json' \
  -d '{"tenant_id":"smoke-cutover","query":"cutover","top_k":5}'
```

Expect a non-empty `memories` array containing the row just written.

## 7. Soak

Leave staging running for **7 days** with normal QA / demo-app traffic.
Monitor:

- `core-storage-api` logs for any `expected N dimensions, not M` errors — should be **none**.
- `core-worker` logs for repeated retries on `EMBED_REQUESTED` — transient OK; sustained means TEI / OpenAI is degraded.
- AlloyDB `SELECT COUNT(*) FROM memories WHERE embedding IS NULL` should trend toward 0 (only newly-written-but-not-yet-embedded rows).
- Spot-check retrieval daily with a known query the QA team uses.

A daily 30-second check is enough; record the numbers in the staging-cutover
tracking issue.

## 8. Done

- Update this runbook's "Last reviewed" header with the date of the
  successful execution.
- Note any deviations from the runbook in the tracking issue (timing
  surprises, additional verification steps, gotchas).
- Open the Spec F (production cutover) PR with this runbook's
  lessons-learned applied. Production must not start until the 7-day
  staging soak has passed without regressions.

## Rollback

If at any section above the verification fails and rollback is needed, see
`docs/runbooks/rollback-010.md` (Spec G). The short version:

1. `gh workflow run migrate-db.yml --repo caura-ai/caura-memclaw-enterprise --ref dev -f target_revision=009 -f destructive=true` — `010`'s `downgrade()` is symmetric (NULLs 1024-dim values, narrows columns back to 768).
2. Roll the core-api / core-storage-api revisions back to the pre-cutover SHA via Cloud Run revision tags.
3. If TEI was wired in section 6, unset the three `OPENAI_EMBEDDING_*` envs to fall back to hosted OpenAI defaults.
4. Re-run backfill against the 768-dim configuration.
