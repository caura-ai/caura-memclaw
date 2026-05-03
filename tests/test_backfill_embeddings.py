"""Unit tests for the OSS embedding-backfill CLI.

The script lives at ``core-storage-api/scripts/backfill_embeddings.py``
and re-embeds rows whose embedding is NULL — the post-migration-010
recovery path for OSS docker-compose users.

These tests mock the engine + ``get_embedding`` so no real DB or
OpenAI account is required. Integration coverage (against a real
local Postgres + fake embedding provider) is covered by the staging
cutover runbook (Spec E), not this PR.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_engine(rows_by_query: dict[str, list[tuple]]) -> MagicMock:
    """Build a minimal AsyncEngine stand-in that returns canned rows for
    each ``conn.execute`` call, keyed by a fragment of the SQL.

    *rows_by_query* maps "memories" / "entities" → the full row list to
    yield in a single page. The first ``execute`` call for each table
    returns those rows; subsequent calls return an empty list (so the
    pagination loop terminates).
    """
    served: dict[str, bool] = {}

    async def _execute(statement, params=None):
        sql = str(statement).lower()
        # UPDATE statements: pretend they succeeded.
        if sql.startswith("update"):
            return MagicMock()
        # SELECT — first call per table returns rows; second returns [].
        for key, rows in rows_by_query.items():
            if key in sql and not served.get(key):
                served[key] = True
                result = MagicMock()
                result.all = MagicMock(return_value=rows)
                return result
        empty = MagicMock()
        empty.all = MagicMock(return_value=[])
        return empty

    conn = MagicMock()
    conn.execute = AsyncMock(side_effect=_execute)
    conn.commit = AsyncMock()

    @asynccontextmanager
    async def _connect():
        yield conn

    engine = MagicMock()
    engine.connect = _connect
    return engine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfill_re_embeds_memories_and_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: each NULL-embedding row gets a re-embed call and a
    corresponding UPDATE. Reports the right scanned/embedded counts."""
    from core_storage_api.scripts.backfill_embeddings import run_backfill

    rows = {
        "from memories": [
            (uuid.uuid4(), "memory content one"),
            (uuid.uuid4(), "memory content two"),
        ],
        "from entities": [
            (uuid.uuid4(), "Acme Corp"),
        ],
    }
    monkeypatch.setattr(
        "core_storage_api.scripts.backfill_embeddings.get_engine"
        if False
        else "core_storage_api.database.init.get_engine",
        lambda: _fake_engine(rows),
    )

    embed = AsyncMock(return_value=[0.1] * 1024)
    monkeypatch.setattr("common.embedding.get_embedding", embed)

    reports = await run_backfill(
        tenant_id=None,
        batch_size=500,
        max_inflight=10,
        dry_run=False,
    )

    by_table = {r.table: r for r in reports}
    assert by_table["memories"].scanned == 2
    assert by_table["memories"].embedded == 2
    assert by_table["entities"].scanned == 1
    assert by_table["entities"].embedded == 1
    # 3 rows → 3 embed calls.
    assert embed.await_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfill_dry_run_skips_provider_and_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dry-run`` counts what would have been done but doesn't call
    the embedding provider or issue UPDATEs."""
    from core_storage_api.scripts.backfill_embeddings import run_backfill

    rows = {
        "from memories": [(uuid.uuid4(), "x"), (uuid.uuid4(), "y")],
        "from entities": [],
    }
    monkeypatch.setattr(
        "core_storage_api.database.init.get_engine",
        lambda: _fake_engine(rows),
    )
    embed = AsyncMock(return_value=[0.1] * 1024)
    monkeypatch.setattr("common.embedding.get_embedding", embed)

    reports = await run_backfill(
        tenant_id=None, batch_size=500, max_inflight=10, dry_run=True
    )

    by_table = {r.table: r for r in reports}
    assert by_table["memories"].scanned == 2
    assert by_table["memories"].embedded == 2  # counted as "would have"
    assert embed.await_count == 0  # but never actually called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfill_skips_empty_content_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row with empty / None content is skipped (not re-embedded with
    a degenerate empty-string vector). Reported under
    ``skipped_empty_content``."""
    from core_storage_api.scripts.backfill_embeddings import run_backfill

    rows = {
        "from memories": [
            (uuid.uuid4(), ""),
            (uuid.uuid4(), "real content"),
            (uuid.uuid4(), None),
        ],
        "from entities": [],
    }
    monkeypatch.setattr(
        "core_storage_api.database.init.get_engine",
        lambda: _fake_engine(rows),
    )
    embed = AsyncMock(return_value=[0.1] * 1024)
    monkeypatch.setattr("common.embedding.get_embedding", embed)

    reports = await run_backfill(
        tenant_id=None, batch_size=500, max_inflight=10, dry_run=False
    )

    mems = next(r for r in reports if r.table == "memories")
    assert mems.scanned == 3
    assert mems.embedded == 1
    assert mems.skipped_empty_content == 2
    assert embed.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfill_aborts_on_consecutive_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``get_embedding`` returns None on too many consecutive rows,
    the backfill raises a RuntimeError so the operator can investigate
    rather than spending the next hour writing nothing."""
    from core_storage_api.scripts import backfill_embeddings

    n_rows = backfill_embeddings._MAX_CONSECUTIVE_NONES + 5
    rows = {
        "from memories": [(uuid.uuid4(), f"content {i}") for i in range(n_rows)],
        "from entities": [],
    }
    monkeypatch.setattr(
        "core_storage_api.database.init.get_engine",
        lambda: _fake_engine(rows),
    )
    embed = AsyncMock(return_value=None)
    monkeypatch.setattr("common.embedding.get_embedding", embed)

    with pytest.raises(RuntimeError, match="consecutive rows"):
        await backfill_embeddings.run_backfill(
            tenant_id=None, batch_size=500, max_inflight=2, dry_run=False
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfill_only_table_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--only-table memories`` skips entities entirely (no scan, no
    report)."""
    from core_storage_api.scripts.backfill_embeddings import run_backfill

    rows = {
        "from memories": [(uuid.uuid4(), "m1")],
        "from entities": [(uuid.uuid4(), "ENTITY-SHOULD-NOT-BE-PROCESSED")],
    }
    monkeypatch.setattr(
        "core_storage_api.database.init.get_engine",
        lambda: _fake_engine(rows),
    )
    embed = AsyncMock(return_value=[0.1] * 1024)
    monkeypatch.setattr("common.embedding.get_embedding", embed)

    reports = await run_backfill(
        tenant_id=None,
        batch_size=500,
        max_inflight=10,
        dry_run=False,
        only_table="memories",
    )

    assert len(reports) == 1
    assert reports[0].table == "memories"


# ---------------------------------------------------------------------------
# CLI exit-code coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_amain_returns_2_on_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RuntimeError`` from ``run_backfill`` (the "consecutive Nones"
    abort path) maps to exit code 2 — distinguishable for monitoring as
    "embedding provider degraded" rather than "config / unexpected"."""
    from core_storage_api.scripts.backfill_embeddings import _amain

    async def _runtime_explode(**_kw):
        raise RuntimeError("provider returned None on 20 consecutive rows; stopping")

    monkeypatch.setattr(
        "core_storage_api.scripts.backfill_embeddings.run_backfill",
        _runtime_explode,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    code = await _amain([])
    assert code == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_amain_returns_1_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Anything that isn't ``RuntimeError`` (DB unreachable,
    registry-level ``ValueError`` surfacing here, asyncio cancellation,
    etc.) maps to exit code 1 with a stack trace logged. Exit-code 1 vs
    2 lets ops monitoring distinguish 'something else is broken' from
    'provider degraded'."""
    import logging

    from core_storage_api.scripts.backfill_embeddings import _amain

    async def _value_explode(**_kw):
        raise ValueError("OPENAI_EMBEDDING_BASE_URL/SEND_DIMENSIONS conflict")

    monkeypatch.setattr(
        "core_storage_api.scripts.backfill_embeddings.run_backfill",
        _value_explode,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with caplog.at_level(
        logging.ERROR, logger="core_storage_api.scripts.backfill_embeddings"
    ):
        code = await _amain([])

    assert code == 1
    assert any(
        "configuration or unexpected error" in rec.getMessage()
        for rec in caplog.records
    ), "expected an ERROR log naming the broader error class"
