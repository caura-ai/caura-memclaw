"""Enrichment constants — moved from ``core_api.constants`` (CAURA-595).

The MEMORY_TYPES / MEMORY_STATUSES tuples and ``DEFAULT_MEMORY_TYPE`` /
``DEFAULT_MEMORY_WEIGHT`` defaults are read by the enrichment service
to validate LLM output. core-api's ``constants.py`` keeps re-exports
for back-compat — adding a new value here should be paired with the
matching pattern + UI updates in core-api.
"""

from __future__ import annotations

# Canonical memory-type vocabulary the enrichment LLM is constrained
# to choose from. Adding a new type requires updating the prompt
# (``common/enrichment/_prompts.py``), the storage schema's CHECK
# constraint, ``MEMORY_TYPES_PATTERN`` in ``core_api.constants``, and
# any UI that switches on type.
MEMORY_TYPES = (
    "fact",
    "episode",
    "decision",
    "preference",
    "task",
    "semantic",
    "intention",
    "plan",
    "commitment",
    "action",
    "outcome",
    "cancellation",
    "rule",
    "insight",
)

DEFAULT_MEMORY_TYPE = "fact"

# Status vocabulary — the enrichment LLM may downgrade a write to e.g.
# ``cancelled`` or ``conflicted`` based on the prompt's classification
# rules.
MEMORY_STATUSES = (
    "active",
    "pending",
    "confirmed",
    "cancelled",
    "outdated",
    "conflicted",
    "archived",
    "deleted",
)

# Default weight assigned to memories whose enrichment didn't return
# a salience score. 0.5 = neutral; the recall ranker uses this as the
# tie-breaking baseline.
DEFAULT_MEMORY_WEIGHT = 0.5
