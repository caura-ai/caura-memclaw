"""Enrichment Pydantic schemas — moved from
``core_api.services.memory_enrichment`` (CAURA-595).

* :class:`AtomicFact` — one self-contained claim broken out of multi-fact
  content.
* :class:`EnrichmentResult` — the full validated enrichment payload the
  service returns. Fields mirror the JSON the LLM is prompted to
  produce in :data:`common.enrichment._prompts.ENRICHMENT_PROMPT`.

Renamed from the legacy ``MemoryEnrichment`` class (kept as a re-export
in ``core_api.services.memory_enrichment``) so that the type's role in
the new event-bus payloads is unambiguous: it's the *result* of
enrichment, distinct from the ``MemoryEnrichRequest`` event payload.
"""

from __future__ import annotations

from pydantic import BaseModel

from common.enrichment.constants import MemoryType


class AtomicFact(BaseModel):
    """One atomic fact extracted from a multi-fact turn.

    Populated by the enricher when a single piece of content carries 2+
    distinct claims that would each be searched by different queries.
    Each fact becomes its own child memory with its own embedding,
    title-less but hint-enriched.
    """

    content: str
    suggested_type: MemoryType = MemoryType.FACT
    retrieval_hint: str = ""


class EnrichmentResult(BaseModel):
    """Validated LLM enrichment output.

    Mirrors the JSON schema in :data:`ENRICHMENT_PROMPT` field-for-field.
    ``llm_ms`` is filled in by the service after the LLM call returns;
    everything else is populated from the LLM's JSON response (with
    defaults for the heuristic fallback path).
    """

    memory_type: MemoryType = MemoryType.FACT
    weight: float = 0.7
    title: str = ""
    summary: str = ""
    tags: list[str] = []
    status: str = "active"
    ts_valid_start: str | None = None
    ts_valid_end: str | None = None
    contains_pii: bool = False
    pii_types: list[str] = []
    retrieval_hint: str = ""
    atomic_facts: list[AtomicFact] | None = None
    llm_ms: int = 0
