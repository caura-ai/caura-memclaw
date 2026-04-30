"""Bump pgvector embedding columns from ``Vector(768)`` to ``Vector(1024)``.

Follows the local-embedder bench-off (``local_emb_res/RESULTS.md``)
that selected ``BAAI/bge-m3`` as the new default embedder. bge-m3 is
natively 1024-dim — running it against the previous 768-dim schema
would require Matryoshka-style truncation that loses retrieval signal
(verified at N=30: native 1024 holds R@5 ≈ 0.96 on LongMemEval
``single-session-user``; truncated 768 trails by ~3 pp in earlier
spot checks).

Affects three columns across two services:
- ``memories.embedding``         — primary memory vectors
- ``entities.name_embedding``    — canonical-name embeddings
- ``documents.embedding``        — opt-in document semantic search

Migration semantics
-------------------
**Existing 768-dim data is destroyed.** pgvector cannot widen a vector
column in place — the dim is part of the column type, and a row's
existing 768-element value is not coercible to 1024. We therefore:

  1. ``DROP INDEX CONCURRENTLY`` the HNSW indexes (so ``ALTER TYPE``
     doesn't have to rebuild them inline). Same pattern used in
     migrations 008/009 for index churn on AlloyDB primary.
  2. ``UPDATE ... SET <embedding column> = NULL`` for every row that
     still has a 768-dim value. Without this, ``ALTER TYPE`` fails
     with ``ERROR: expected 1024 dimensions, not 768``. Setting to
     NULL is acceptable: pgvector treats NULL as "no embedding",
     queries naturally exclude these rows from semantic search, and
     the application re-embeds on next write.
  3. ``ALTER TABLE ... ALTER COLUMN ... TYPE vector(1024)`` —
     succeeds now that the column is empty.
  4. ``CREATE INDEX CONCURRENTLY`` the HNSW indexes against the new
     1024-dim column. Same operator (``vector_cosine_ops``) and build
     params (``m = 16, ef_construction = 64``) as the originals in
     001 / 003.

After upgrade, all existing memories/entities/documents have NULL
embeddings until the application re-embeds them. Two paths to recover:
- **Lazy**: any new write or any read path that calls
  ``get_or_cache_embedding`` will re-embed the row's content via the
  configured TEI sidecar. Steady-state OSS deployments converge over
  hours/days as memories are touched.
- **Eager**: run the backfill task in ``core-worker`` (separate PR)
  to scan rows ``WHERE embedding IS NULL`` and embed in batches.
  Recommended for production cutovers.

Operationally
-------------
- ``CONCURRENTLY`` keeps reads/writes serving through the index churn
  on AlloyDB primary — same constraint that drove migrations 008/009.
- The ``ALTER TYPE`` itself takes a brief ``ACCESS EXCLUSIVE`` lock on
  each table. On a multi-million-row table this is sub-second once
  the column is empty (the lock is metadata-only, no row rewrites).
- ``UPDATE ... = NULL`` does rewrite every affected row — long-running
  on production-sized tables, but no exclusive lock and no app
  downtime. Safe to run during traffic.

Downgrade
---------
Symmetric: drop indexes, NULL the 1024-dim values, ALTER TYPE back to
``Vector(768)``, recreate indexes. Same data-loss tradeoff in reverse.

Revision ID: 010
Revises: 009
Create Date: 2026-04-30
"""

from collections.abc import Sequence

from alembic import op

revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Drop HNSW indexes — must be CONCURRENTLY (read traffic stays
    # served on AlloyDB primary). ``CONCURRENTLY`` cannot run inside a
    # transaction, so we use alembic's autocommit_block. Same idiom as
    # migrations 008 / 009.
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_memories_embedding_hnsw"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_documents_embedding_hnsw"
        )
    # ``entities.name_embedding`` has no HNSW index (per 001) — no-op.

    # 2. Clear existing 768-dim values so ``ALTER TYPE`` accepts the
    # new dim. NULL is the only value compatible with both old and
    # new column types. Application is expected to re-embed
    # post-migration (see module docstring).
    op.execute(
        "UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL"
    )
    op.execute(
        "UPDATE entities SET name_embedding = NULL "
        "WHERE name_embedding IS NOT NULL"
    )
    op.execute(
        "UPDATE documents SET embedding = NULL WHERE embedding IS NOT NULL"
    )

    # 3. Widen each column to 1024 dim. Sub-second metadata-only
    # change once the column is empty.
    op.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1024)")
    op.execute(
        "ALTER TABLE entities ALTER COLUMN name_embedding TYPE vector(1024)"
    )
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1024)"
    )

    # 4. Rebuild HNSW indexes against the new dim. Same operator + build
    # params as the originals in 001 / 003.
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memories_embedding_hnsw
            ON memories
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_embedding_hnsw
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            WHERE embedding IS NOT NULL
            """
        )


def downgrade() -> None:
    # Mirror image of upgrade. Same data-loss tradeoff: existing
    # 1024-dim embeddings cannot be coerced back to 768; rows are
    # NULLed and the application is expected to re-embed against
    # whatever 768-dim provider it switches back to.
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_memories_embedding_hnsw"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_documents_embedding_hnsw"
        )

    op.execute(
        "UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL"
    )
    op.execute(
        "UPDATE entities SET name_embedding = NULL "
        "WHERE name_embedding IS NOT NULL"
    )
    op.execute(
        "UPDATE documents SET embedding = NULL WHERE embedding IS NOT NULL"
    )

    op.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768)")
    op.execute(
        "ALTER TABLE entities ALTER COLUMN name_embedding TYPE vector(768)"
    )
    op.execute(
        "ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768)"
    )

    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_memories_embedding_hnsw
            ON memories
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_embedding_hnsw
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            WHERE embedding IS NOT NULL
            """
        )
