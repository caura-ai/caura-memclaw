"""Embedding retry with graceful degradation tests.

Unit tests validate:
  - Embedding retry constants are sensible
  - Retry exhaustion returns None instead of raising
  - Success on second attempt returns valid embedding
  - Fake provider never returns None (no retry needed)
"""

from unittest.mock import AsyncMock, patch

import pytest

from core_api.constants import (
    EMBEDDING_REEMBED_BATCH_SIZE,
    EMBEDDING_REEMBED_DELAY_S,
    EMBEDDING_RETRY_ATTEMPTS,
    EMBEDDING_RETRY_DELAY_S,
)


# ---------------------------------------------------------------------------
# Unit tests: constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbeddingRetryConstants:
    """Verify embedding retry constants are sensible."""

    def test_retry_attempts_positive(self):
        assert EMBEDDING_RETRY_ATTEMPTS >= 1

    def test_retry_attempts_bounded(self):
        assert EMBEDDING_RETRY_ATTEMPTS <= 5

    def test_retry_delay_positive(self):
        assert EMBEDDING_RETRY_DELAY_S > 0

    def test_retry_delay_bounded(self):
        worst = sum(
            EMBEDDING_RETRY_DELAY_S * (i + 1) for i in range(EMBEDDING_RETRY_ATTEMPTS)
        )
        assert worst <= 30.0

    def test_reembed_delay_positive(self):
        assert EMBEDDING_REEMBED_DELAY_S > 0

    def test_reembed_batch_size_positive(self):
        assert EMBEDDING_REEMBED_BATCH_SIZE >= 1


# ---------------------------------------------------------------------------
# Unit tests: retry exhaustion returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retry_exhaustion_returns_none():
    """When all retry attempts fail, get_embedding returns None."""
    from common.embedding import get_embedding

    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(side_effect=RuntimeError("provider down"))
    with (
        patch(
            "common.embedding._service.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch("common.embedding._service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await get_embedding("hello world")

    assert result is None
    assert mock_provider.embed.call_count == EMBEDDING_RETRY_ATTEMPTS


# ---------------------------------------------------------------------------
# Unit tests: success on second attempt
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_success_on_second_attempt():
    """get_embedding returns a valid embedding when the second attempt succeeds."""
    from common.embedding import get_embedding

    fake_vec = [0.1] * 768
    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(side_effect=[RuntimeError("transient"), fake_vec])
    with (
        patch(
            "common.embedding._service.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch("common.embedding._service.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await get_embedding("hello world")

    assert result == fake_vec
    assert mock_provider.embed.call_count == 2


# ---------------------------------------------------------------------------
# Unit tests: fake provider never returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fake_provider_never_returns_none():
    """The fake embedding provider always succeeds (deterministic hash)."""
    from common.embedding import get_embedding

    # CAURA-594 extraction: common/embedding/_service reads EMBEDDING_PROVIDER
    # from os.environ when no tenant_config is passed. Force "fake" via the
    # env var rather than patching a now-nonexistent module attribute.
    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.setenv("EMBEDDING_PROVIDER", "fake")
    try:
        result = await get_embedding("any text")
    finally:
        monkeypatch_env.undo()

    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
