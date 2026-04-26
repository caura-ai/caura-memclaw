"""P2-6: Enrichment prompt signal phrases for better type classification.

Unit tests validate:
- Prompt text includes signal phrases for decision, commitment, rule
- Signal phrases don't inflate prompt token count excessively
- Keyword heuristic still correctly classifies all types (regression)
"""

import pytest


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEnrichmentPromptSignalPhrases:
    """Verify signal phrases are present in the enrichment prompt."""

    def _get_prompt(self) -> str:
        from core_api.services.memory_enrichment import ENRICHMENT_PROMPT
        return ENRICHMENT_PROMPT

    def test_decision_has_signal_phrases(self):
        prompt = self._get_prompt()
        assert "decided to" in prompt
        assert "going with" in prompt
        assert "opted for" in prompt

    def test_commitment_has_signal_phrases(self):
        prompt = self._get_prompt()
        assert "committed to" in prompt
        assert "pledged" in prompt

    def test_rule_has_signal_phrases(self):
        prompt = self._get_prompt()
        # Rule line should mention key directive words
        assert "always" in prompt.lower()
        assert "never" in prompt.lower()

    def test_prompt_token_count_reasonable(self):
        """Prompt ceiling — keeps cost bounded while leaving room for small additions.

        Baseline grew with retrieval_hint, atomic_facts, and the temporal
        ts_valid_end guidance; 1100 gives ~15% headroom over today's ~942.
        """
        prompt = self._get_prompt()
        word_count = len(prompt.split())
        assert word_count < 1100, f"Prompt is {word_count} words — too long, will increase cost"

    def test_prompt_structure_unchanged(self):
        """JSON template should still be present and valid."""
        prompt = self._get_prompt()
        assert '"memory_type": "..."' in prompt
        assert '"weight": 0.0' in prompt
        assert '"tags": ["..."]' in prompt
        assert "Content:" in prompt


@pytest.mark.unit
class TestHeuristicRegressions:
    """Ensure keyword heuristic still classifies all types correctly after prompt changes."""

    def test_decision_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("We decided to use PostgreSQL").memory_type == "decision"
        assert _fake_enrich("Team chose Redis for caching").memory_type == "decision"
        assert _fake_enrich("Management approved the budget").memory_type == "decision"

    def test_preference_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("The team prefers dark mode").memory_type == "preference"

    def test_episode_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("We deployed v2.3 to production").memory_type == "episode"

    def test_task_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("Need to review the PR by Friday").memory_type == "task"

    def test_commitment_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("We committed to delivering by Q2").memory_type == "commitment"

    def test_rule_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("Always notify security before deploying").memory_type == "rule"
        assert _fake_enrich("Never store PII in Redis").memory_type == "rule"

    def test_cancellation_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("The project was cancelled last week").memory_type == "cancellation"

    def test_outcome_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("The migration achieved 99.9% uptime").memory_type == "outcome"

    def test_fact_default(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("The server runs on port 8080").memory_type == "fact"

    def test_plan_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("The roadmap includes three phases").memory_type == "plan"

    def test_intention_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("We intend to migrate to AWS next quarter").memory_type == "intention"

    def test_action_classification(self):
        from core_api.services.memory_enrichment import _fake_enrich
        assert _fake_enrich("I executed the database backup script").memory_type == "action"
