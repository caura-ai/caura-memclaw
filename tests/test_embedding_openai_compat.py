"""Unit tests for the A.5 + Option B knobs on ``OpenAIEmbeddingProvider``.

These verify the contract added in the local-embedder branch — the
provider now also serves any OpenAI-compatible endpoint (TEI sidecar,
vLLM, etc.) and supports asymmetric query/document encoding for
instruction-aware models.

What's covered (no network, no real OpenAI / TEI calls):

* ``base_url`` is forwarded to the underlying OpenAI SDK.
* ``send_dimensions=False`` omits the ``dimensions`` SDK kwarg
  (TEI rejects it).
* ``send_dimensions=True`` (default) still sends ``dimensions=VECTOR_DIM``
  on the OpenAI hosted path.
* ``truncate_to_dim`` slices and L2-renormalizes the model output.
* ``embed_query`` is identical to ``embed`` when no instruction
  resolves (symmetric models like bge-m3, OpenAI text-embedding-3-small).
* ``embed_query`` prepends the Qwen-style ``Instruct: ...\\nQuery: ...``
  prefix when an instruction resolves (per-call arg or constructor
  default).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.constants import VECTOR_DIM
from common.embedding.providers.openai import OpenAIEmbeddingProvider


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _fake_openai_response(vector: list[float]):
    """Shape an OpenAI SDK ``embeddings.create`` response object."""
    return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=vector)])


def _patched_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response_vector: list[float] | None = None,
    **provider_kwargs,
) -> tuple[OpenAIEmbeddingProvider, MagicMock, AsyncMock]:
    """Build an ``OpenAIEmbeddingProvider`` whose underlying
    ``openai.AsyncOpenAI`` is fully mocked.

    Returns ``(provider, async_openai_constructor_mock, embeddings_create_mock)``
    so each test can assert against either the constructor kwargs (was
    ``base_url`` forwarded?) or the create-call kwargs (was ``dimensions``
    sent? was the input prefixed?).
    """
    if response_vector is None:
        response_vector = [0.5] * VECTOR_DIM

    embeddings_create = AsyncMock(return_value=_fake_openai_response(response_vector))
    fake_client = MagicMock()
    fake_client.embeddings.create = embeddings_create

    async_openai_ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(
        "common.embedding.providers.openai.openai.AsyncOpenAI", async_openai_ctor
    )

    provider = OpenAIEmbeddingProvider(api_key="sk-fake", **provider_kwargs)
    return provider, async_openai_ctor, embeddings_create


# ---------------------------------------------------------------------------
# A.5 — base_url forwarding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_base_url_forwarded_to_async_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """``base_url`` is passed through to ``openai.AsyncOpenAI`` so a TEI
    sidecar (``http://tei:80/v1``) becomes a drop-in OpenAI replacement."""
    _, async_openai_ctor, _ = _patched_provider(
        monkeypatch, base_url="http://tei:80/v1"
    )
    kwargs = async_openai_ctor.call_args.kwargs
    assert kwargs.get("base_url") == "http://tei:80/v1"


@pytest.mark.unit
def test_base_url_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``base_url`` is not given, the kwarg must not be sent
    (would override the SDK's hosted-OpenAI default)."""
    _, async_openai_ctor, _ = _patched_provider(monkeypatch)
    assert "base_url" not in async_openai_ctor.call_args.kwargs


# ---------------------------------------------------------------------------
# A.5 — send_dimensions toggle
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_dimensions_false_omits_dimensions_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TEI rejects ``dimensions=`` — when pointed at a TEI sidecar,
    ``send_dimensions=False`` must drop the kwarg from every embed call."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=False)
    await provider.embed("hello")
    assert "dimensions" not in create.call_args.kwargs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_dimensions_true_sends_vector_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default path (hosted OpenAI) still sends the ``dimensions``
    kwarg pinned to the schema's ``VECTOR_DIM``."""
    provider, _, create = _patched_provider(monkeypatch)  # default True
    await provider.embed("hello")
    assert create.call_args.kwargs.get("dimensions") == VECTOR_DIM


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_respects_send_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed_batch`` honours the same toggle as ``embed`` —
    important so a TEI sidecar serves both ingest and search paths."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=False)
    create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=0, embedding=[0.1] * VECTOR_DIM),
            SimpleNamespace(index=1, embedding=[0.2] * VECTOR_DIM),
        ]
    )
    await provider.embed_batch(["a", "b"])
    assert "dimensions" not in create.call_args.kwargs


# ---------------------------------------------------------------------------
# Option B — Matryoshka truncation + L2 renormalization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_truncate_to_dim_slices_and_renormalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vectors trained with MRL (Qwen3-Embedding, jina-v3,
    snowflake-arctic-l) stay coherent under truncation, but the
    truncated vector is no longer unit-norm. The provider must slice
    AND L2-renormalize so cosine similarity at the pgvector layer
    stays correct."""
    raw = list(range(1, 11))  # native dim 10 → truncate to 4
    provider, _, _ = _patched_provider(
        monkeypatch, response_vector=raw, truncate_to_dim=4, send_dimensions=False
    )
    out = await provider.embed("anything")

    assert len(out) == 4
    norm = math.sqrt(sum(x * x for x in out))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)
    # Verify proportional structure preserved (the slice is [1,2,3,4]
    # → renormalised; relative ratios survive).
    expected = [v / math.sqrt(1 + 4 + 9 + 16) for v in (1, 2, 3, 4)]
    for got, exp in zip(out, expected, strict=True):
        assert math.isclose(got, exp, abs_tol=1e-9)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_truncate_to_dim_unset_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No truncation knob → vector is returned untouched
    (no surprise renormalisation on the OpenAI hosted path)."""
    raw = [0.3, 0.4, 0.0, 0.0]  # not unit-norm
    provider, _, _ = _patched_provider(
        monkeypatch, response_vector=raw, send_dimensions=False
    )
    out = await provider.embed("hi")
    assert out == raw


# ---------------------------------------------------------------------------
# Option B — embed_query (asymmetric, instruction-aware)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_query_no_instruction_passes_text_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric models (bge-m3, gte-en-v1.5, OpenAI
    text-embedding-3-small) get no prefix — ``embed_query`` is
    equivalent to ``embed`` when no instruction resolves."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=False)
    await provider.embed_query("what is the capital of France?")
    assert create.call_args.kwargs["input"] == "what is the capital of France?"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_query_per_call_instruction_prepended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call ``instruction`` arg overrides the constructor default
    and produces the Qwen-style ``Instruct: ...\\nQuery: ...`` prefix."""
    provider, _, create = _patched_provider(monkeypatch, send_dimensions=False)
    await provider.embed_query("paris", instruction="Retrieve relevant memories")
    expected = "Instruct: Retrieve relevant memories\nQuery: paris"
    assert create.call_args.kwargs["input"] == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_query_constructor_default_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``query_instruction=`` set at construction is the fallback
    for any caller that doesn't override per-call. This is the
    process-wide default plumbed via ``EMBEDDING_QUERY_INSTRUCTION``."""
    provider, _, create = _patched_provider(
        monkeypatch,
        send_dimensions=False,
        query_instruction="Default task",
    )
    await provider.embed_query("query body")
    assert create.call_args.kwargs["input"] == "Instruct: Default task\nQuery: query body"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_unaffected_by_query_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``embed`` (the document/ingest path) NEVER prepends the
    instruction — that asymmetry is the whole point of Option B."""
    provider, _, create = _patched_provider(
        monkeypatch,
        send_dimensions=False,
        query_instruction="Should not appear",
    )
    await provider.embed("a document chunk")
    assert create.call_args.kwargs["input"] == "a document chunk"
