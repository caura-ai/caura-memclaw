# Local Embedder (TEI sidecar)

MemClaw can serve embeddings from a self-hosted [HuggingFace Text Embeddings
Inference](https://github.com/huggingface/text-embeddings-inference) (TEI)
container instead of OpenAI's hosted API. **Default model: `BAAI/bge-m3`**
(1024-dim, MIT-licensed, multilingual, no instruction prefix needed) — matches
the schema set in alembic migration `012_vector_dim_1024`.

## TL;DR

```bash
docker compose --profile embed-local up -d
```

That's it. The `tei` service runs on the compose network, `core-api` finds it
at `http://tei:80/v1`, embeddings stop hitting OpenAI. Set the four envs from
`.env.example` if you want to override the model or point at an external TEI.

## Why bother

| | OpenAI hosted | TEI + bge-m3 (this) |
|---|---|---|
| Cost @ 1000 RPS | ~$10.5k/mo | ~$1.0k/mo (2× L4) |
| Idle p50 search | ~230 ms | ~127 ms (CPU), ~50 ms (L4) |
| Data residency | leaves your env | stays in your VPC |
| R@5 (LongMemEval N=30) | 0.889 | 0.963 |
| Accuracy | 86.7 % | 93.3 % |

Numbers from `local_emb_res/RESULTS.md` (the bench-off that drove this default).

## How it works

`OpenAIEmbeddingProvider` accepts an optional `base_url`. When set, the OpenAI
SDK routes every embed call there instead of `api.openai.com`. TEI speaks the
same `POST /v1/embeddings` API shape, so no other code changes are required.

Two flags are TEI-specific:

* **`OPENAI_EMBEDDING_SEND_DIMENSIONS=false`** — TEI rejects the
  `dimensions=` SDK kwarg. Must be `false` whenever `OPENAI_EMBEDDING_BASE_URL`
  points at TEI. The OpenAI hosted path keeps the default `true` (dim is
  pinned to `VECTOR_DIM`).
* **`OPENAI_EMBEDDING_TRUNCATE_TO_DIM`** — Matryoshka-style output slice +
  L2-renormalize. Only relevant if you swap to a model whose native dim is
  larger than the schema (e.g. Qwen3-Embedding-0.6B = 1024, schema = 768
  on a stale deployment). bge-m3 is native 1024 against the 1024 schema —
  leave empty.

Two are model-specific:

* **`OPENAI_EMBEDDING_MODEL`** — passed to the OpenAI SDK and surfaced to TEI
  unchanged. TEI ignores it (TEI runs whatever model `--model-id` says) but
  the value is logged and cached on so we keep it accurate.
* **`EMBEDDING_QUERY_INSTRUCTION`** — only set this for instruction-aware
  models (Qwen3-Embedding, e5-instruct, KaLM). When set, the search path
  prepends `Instruct: <task>\nQuery: <text>` to every query; the ingest path
  is unchanged. bge-m3 is symmetric — leave empty.

The full asymmetric-encoding contract is in `common/embedding/protocols.py`
(`embed_query` vs `embed`).

## Hardware sizing

The compose service ships with the **CPU image** (`cpu-1.7`) by default. Good
enough for a laptop, a small VM, or a single-tenant client. Numbers below are
for `BAAI/bge-m3` at ~200-token average input.

| Hardware | TEI image | Realistic RPS | p50 latency |
|---|---|---|---|
| 8 vCPU AMD (e2-standard-8) | `cpu-1.7` (default) | 200–400 | ~130 ms |
| 8 vCPU Intel (n2-standard-8, AVX-512) | `cpu-1.7` | 400–700 | ~80 ms |
| 1× L4 GPU (g2-standard-8) | `89-1.7` | 3,000–8,000 | ~50 ms |

To switch to the GPU image, set `TEI_IMAGE_TAG=89-1.7` in `.env` **and**
uncomment the `deploy.resources.reservations.devices` block in the `tei`
service (or copy it into `docker-compose.override.yml`). You'll need
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on the host.

For production at sustained high RPS, you want **two TEI replicas behind a
load balancer** — see the discussion in `local_emb_res/sizing-1000rps.md`.
A single replica's per-call latency degrades sharply under concurrent load
(measured: 57 ms idle → 571 ms at 5×2 concurrent fan-out).

## Swapping models

Set `TEI_MODEL_ID` and (if needed) the four flags above. The combinations
that work cleanly with the current 1024-dim schema:

| Model | Native dim | Truncate? | Instruction? | Notes |
|---|---|---|---|---|
| `BAAI/bge-m3` | 1024 | no | no | **Default.** MIT, 100+ langs, 8K ctx. |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1024 | no | no (recommended `query:` prefix) | Apache-2.0, 74 langs. Same shape as bge-m3. |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | no | yes | Apache-2.0, instruction-aware. Best quality for this dim, but needs GPU. |
| `Qwen/Qwen3-Embedding-4B` | 2560 | yes (`=1024`) | yes | Truncates to 1024 via Matryoshka. L4 GPU. |

If you go to a model with a different native dim (e.g. Qwen3-8B at 4096),
either set `OPENAI_EMBEDDING_TRUNCATE_TO_DIM=1024` or migrate the schema to
the new dim with a follow-up alembic revision.

## Schema requirement

> **Upgrading from v1.x?** Follow the
> [Upgrading from v1.x](../README.md#upgrading-from-v1x) section in the
> README first — it sequences DB snapshot, opt-in env var, and re-embed for
> you. The bare `alembic upgrade 012` shown below is for fresh installations.

The `tei` profile assumes the 1024-dim schema. If you're upgrading an existing
768-dim deployment:

```bash
# Pre-flight first — read-only readout of how many rows will be NULL'd,
# rough wall-clock estimate, and the exact opt-in command to run next.
docker compose run --rm core-storage-api \
  python -m core_storage_api.scripts.preflight_012

# Run the migration (NULLs existing 768-dim embeddings; re-embed required)
docker compose run --rm core-storage-api \
  python -m alembic upgrade 012
```

The `012` revision is **destructive** — existing 768-dim vectors cannot be
coerced to 1024 and are set to `NULL`. The application re-embeds lazily on
next read/write. For an eager re-embed pass, run the bundled CLI:

```bash
docker compose run --rm core-storage-api \
  python -m core_storage_api.scripts.backfill_embeddings --dry-run
docker compose run --rm core-storage-api \
  python -m core_storage_api.scripts.backfill_embeddings
```

For multi-tenant production cutovers prefer the event-driven backfill task
in `core-worker`. It scans `WHERE embedding IS NULL` and publishes one
`EMBED_REQUESTED` event per row, inheriting the consumer's per-tenant
concurrency, retry, and DLQ wiring:

```bash
docker compose run --rm core-worker \
  python -m core_worker.cli backfill-embeddings --dry-run
docker compose run --rm core-worker \
  python -m core_worker.cli backfill-embeddings
```

Optional knobs: `--tenant-id <id>` (per-tenant cutover), `--batch-size N`,
`--max-inflight N`, `--dry-run`. Idempotent and restartable — re-running
picks up only rows still missing an embedding.

See `core-storage-api/src/core_storage_api/database/migrations/versions/012_vector_dim_1024.py`
for the full operational story.

## Switching back to OpenAI hosted

Just don't pass `--profile embed-local`. The `tei` service is profile-gated,
so a plain `docker compose up -d` brings up the same stack as before — no
TEI container, `core-api` falls through to the hosted OpenAI default
(unset `OPENAI_EMBEDDING_BASE_URL`, set `OPENAI_API_KEY`).

A single deployment cannot mix both — the schema dim is global. To go back
to OpenAI's `text-embedding-3-small` (1536 → 768 truncated), you'd need to
roll back migration `012` and re-embed.

## Troubleshooting

* **`tei` healthcheck fails on first boot for several minutes.** First-time
  model download can be slow (~2 GB for bge-m3). The healthcheck has 30
  retries and `start_period: 60s` to cover this. Subsequent boots reuse the
  `tei_cache` volume and come up in seconds.
* **`core-api` logs `400 Bad Request` from `/v1/embeddings`.** Check that
  `OPENAI_EMBEDDING_SEND_DIMENSIONS=false` — TEI rejects the `dimensions=`
  kwarg.
* **Per-call latency jumps from ~130 ms to ~600 ms under concurrent load.**
  Single-replica TEI queues requests in dynamic batching. Add a second
  replica behind nginx/LB — see `local_emb_res/design_integration.md`.
* **`413 Payload Too Large` on long writes.** The compose `command:` includes
  `--auto-truncate` so this should not happen with bge-m3 (8K context).
  If it does, you've swapped to a 512-ctx model — pick something else or
  pre-chunk in the application layer.
