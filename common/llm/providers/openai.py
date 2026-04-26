"""OpenAI-compatible LLM provider — moved from
``core_api.providers.openai_provider`` (CAURA-595).

Wraps the ``openai`` SDK (AsyncOpenAI) to implement the
``LLMProvider`` protocol. Supports OpenAI, Anthropic (via OpenAI-
compatible endpoint), and OpenRouter by varying the ``base_url``
parameter.

The previous ``settings.openai_request_timeout_seconds`` import has
been replaced with a constructor arg defaulting to
``OPENAI_REQUEST_TIMEOUT_SECONDS`` from ``common.llm.constants`` —
the registry passes the resolved value through. Same decoupling
shape Step B used for ``OpenAIEmbeddingProvider``.
"""

from __future__ import annotations

import json
import logging
import time

import openai

from common.llm.constants import OPENAI_CHAT_BASE_URL, OPENAI_REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class OpenAILLMProvider:
    """LLM provider using the OpenAI chat completions API.

    Works with any OpenAI-compatible endpoint (OpenAI, Anthropic, OpenRouter)
    by setting the appropriate ``base_url``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = OPENAI_CHAT_BASE_URL,
        provider_name: str = "openai",
        request_timeout_seconds: float = OPENAI_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._provider_name = provider_name
        # Explicit per-call timeout — without this the SDK rides httpx's
        # default and a single hung upstream call would eat the whole
        # enrichment budget silently.
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    async def complete_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
    ) -> dict:
        """Send a prompt and return a parsed JSON dict.

        Uses ``response_format={"type": "json_object"}`` to enforce JSON output.
        """
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "OpenAI-compatible complete_json (%s) took %dms",
            self._model,
            llm_ms,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"OpenAI returned empty content for model {self._model}")
        return json.loads(content)

    async def complete_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Send a prompt and return the raw text content."""
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "OpenAI-compatible complete_text (%s) took %dms",
            self._model,
            llm_ms,
        )
        return response.choices[0].message.content or ""
