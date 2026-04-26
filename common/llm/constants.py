"""LLM-provider constants — moved from ``core_api.constants`` (CAURA-595).

Mirrors ``common/embedding/constants.py``. core-api's ``constants.py``
keeps re-exports for back-compat; new code should import from here.
"""

from __future__ import annotations

import os

# ── Provider model defaults ──────────────────────────────────────────

VERTEX_LLM_DEFAULT_MODEL = "gemini-2.0-flash-lite"
GEMINI_DEFAULT_MODEL = os.environ.get(
    "GEMINI_DEFAULT_MODEL", "gemini-3.1-flash-lite-preview"
)

# OpenAI's chat-completions base URL — used by ``OpenAILLMProvider``
# (and for openrouter / anthropic compat where the API mirrors OpenAI's
# shape, with the base URL swapped via tenant-config override).
OPENAI_CHAT_BASE_URL = "https://api.openai.com/v1"

# Anthropic + OpenRouter base URLs and default models. The
# ``OpenAILLMProvider`` works against any of these endpoints by varying
# ``base_url``; the registry picks the right tuple based on
# ``ProviderName``.
ANTHROPIC_CHAT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_DEFAULT_MODEL = os.environ.get(
    "ANTHROPIC_DEFAULT_MODEL", "claude-haiku-4-5-20251001"
)  # Anthropic API requires native model IDs
OPENROUTER_CHAT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_DEFAULT_MODEL", "openai/gpt-5.4-nano"
)

# ── Retry policy ─────────────────────────────────────────────────────

# Retries on primary provider before falling back to a secondary
# provider (configured via ``call_with_fallback``). Linear backoff
# rather than exponential because the LLM call is already slow (1-3s)
# and a multi-second backoff would push the request past timeout.
LLM_RETRY_ATTEMPTS = 2
LLM_RETRY_DELAY_S = 1.0

# Fallback model for OpenAI-compatible providers when the tenant's
# configured model is not set — env-overridable so on-call can swap to
# a cheaper / different family without a redeploy.
LLM_FALLBACK_MODEL_OPENAI = os.environ.get("LLM_FALLBACK_MODEL_OPENAI", "gpt-5.4-nano")

# Per-call timeout passed to the OpenAI/Anthropic/Openrouter SDK.
# Without an explicit value the SDK rides httpx's default — long
# enough that a single hung upstream call eats the whole enrichment
# budget silently. 25s gives the provider room to respond while still
# leaving budget for one retry under the inline ceiling.
def _read_openai_request_timeout_seconds() -> float:
    """Parse ``OPENAI_REQUEST_TIMEOUT_SECONDS`` defensively.

    Bare ``float(os.environ.get(...))`` would raise ``ValueError`` at
    module import on a misconfigured value (e.g. ``"25s"`` instead of
    ``"25"``), crashing the entire worker / core-api startup before any
    structured-logging is wired. Catch it here, write a warning to
    stderr, and fall back to the documented default.
    """
    raw = os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "25.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        # Module-level import has no logger configured yet — print to
        # stderr with the key name so the bad value is visible in the
        # crash log even before structlog wires up.
        import sys

        print(
            f"WARN: OPENAI_REQUEST_TIMEOUT_SECONDS={raw!r} is not a "
            f"valid float; falling back to 25.0",
            file=sys.stderr,
        )
        return 25.0


OPENAI_REQUEST_TIMEOUT_SECONDS = _read_openai_request_timeout_seconds()
