"""Swappable LLM provider interface protocol and client implementations.

Both providers talk to their vendor's plain REST endpoint over ``httpx`` rather
than the ``groq``/``google-generativeai`` SDKs. This is a deliberate simplification,
not a missing feature: each API is one HTTP POST and a JSON body, ``httpx`` is
already a project dependency, and skipping the SDKs means selecting a provider
never requires an extra install. Both are the vendors' documented **free-tier**
endpoints — no paid subscription, matching the brief's constraint.

Every call site in this codebase (:func:`app.agents.seo_agent.apply_llm_suggested_fixes`,
:func:`app.agents.nap_agent.filter_candidates_with_llm`,
:func:`app.agents.qa_agent.answer_question`) already treats a raising ``generate``
identically to no LLM at all — catches the exception and falls back to the fully
deterministic result. So these providers raise plain, informative exceptions
(missing key, HTTP error, unparseable response) rather than swallowing anything
themselves: masking the failure here would only hide it from whoever is debugging
why suggestions look unpolished, while the pipeline's correctness never depends on
it.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

#: Per-request timeout. A hung LLM call must not hang the whole audit.
DEFAULT_TIMEOUT = 30.0

#: Free-tier default models. Overridable via the constructor or an env var
#: (``GROQ_MODEL`` / ``GEMINI_MODEL``) without touching code, matching the plan's
#: "swapping the .env value changes provider with zero code changes" goal.
#:
#: Both verified via web search as of September 2026, not carried over from
#: training data — model IDs on both platforms rot fast enough that doing
#: otherwise would ship a dead default. The originally-shipped defaults
#: (``llama-3.1-8b-instant``, ``gemini-1.5-flash``) were checked the same way and
#: were both already stale: Groq deprecated ``llama-3.1-8b-instant`` in June 2026
#: (Groq's own recommended replacement is ``openai/gpt-oss-20b``, used below), and
#: every ``gemini-1.5-*`` model now 404s — Gemini 1.5 was fully shut down.
#:
#: ``gemini-flash-latest`` is deliberately an alias, not a pinned version: Google
#: hot-swaps it to the current Flash model on every release (with a 2-week notice
#: for breaking changes), which is the direct fix for the exact staleness that bit
#: the old pinned default. Still verify both before a real demo — an assistant's
#: knowledge of "current" model IDs is never a substitute for checking on the day.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


class LLMProvider(ABC):
    """Abstract base class interface for swappable LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion from the LLM provider for a given prompt."""
        pass


class ProviderConfigError(RuntimeError):
    """Raised when a provider is selected but not usably configured (no API key)."""


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation.

    Talks to Groq's OpenAI-compatible chat completions endpoint — the same free
    tier the plan names.
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            api_key: Groq API key. Falls back to the ``GROQ_API_KEY`` env var,
                then ``LLM_API_KEY`` (the name used in ``.env.example`` when
                ``LLM_PROVIDER=groq``).
            model: Model id. Falls back to ``GROQ_MODEL``, then
                :data:`DEFAULT_GROQ_MODEL`.
            timeout: Per-request timeout in seconds.
            client: Optional pre-built ``httpx.Client`` — tests substitute one
                with a mock transport instead of touching the network.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
        self.timeout = timeout
        self._client = client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion using the Groq API.

        Args:
            prompt: The prompt text.
            **kwargs: ``temperature`` and ``max_tokens`` are forwarded if given;
                anything else is ignored so callers can pass through the same
                keyword set used elsewhere without this provider breaking.
                ``reasoning_effort`` ("low"/"medium"/"high"), if given, is
                forwarded to Groq as-is — see the note below.

        Returns:
            The completion text, stripped of leading/trailing whitespace.

        Raises:
            ProviderConfigError: No API key is configured.
            httpx.HTTPStatusError: The API returned a non-2xx response.
            ValueError: The response body was not the expected shape.

        Note on ``reasoning_effort``:
            :data:`DEFAULT_GROQ_MODEL` (``openai/gpt-oss-20b``) is a *reasoning*
            model — before writing its final answer it writes an internal
            ``reasoning`` field, and that reasoning draws from the SAME
            ``max_tokens`` budget as the answer. Found live: judging a realistic
            15-passage answerability prompt, this model sometimes reasons in
            circles ("But... but... but...") and burns the entire budget without
            ever reaching a conclusion — confirmed directly (``finish_reason`` ==
            ``"length"``, ~1600 reasoning tokens used, 0 content) even at a
            generous ``max_tokens=1600``. Raising the budget further is not a
            reliable fix, since the model can keep expanding to fill whatever
            budget it is given. Passing ``reasoning_effort="low"`` is: on the
            identical prompt it converged to a correct, verified answer in ~39
            reasoning tokens, reliably across 5 repeated live calls. Only
            forwarded when the caller supplies it, so callers on non-reasoning
            Groq models are unaffected.
        """
        if not self.api_key:
            raise ProviderConfigError(
                "Groq API key not configured — set GROQ_API_KEY or LLM_API_KEY."
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        if kwargs.get("reasoning_effort"):
            payload["reasoning_effort"] = kwargs["reasoning_effort"]

        data = _post_json(
            self._client,
            self.API_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout=self.timeout,
        )

        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(f"Unexpected Groq response shape: {data!r}") from exc


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation.

    Talks to the Gemini free-tier REST endpoint (``generativelanguage.googleapis.com``).
    """

    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            api_key: Gemini API key. Falls back to ``GEMINI_API_KEY``, then
                ``LLM_API_KEY``.
            model: Model id. Falls back to ``GEMINI_MODEL``, then
                :data:`DEFAULT_GEMINI_MODEL`.
            timeout: Per-request timeout in seconds.
            client: Optional pre-built ``httpx.Client`` — tests substitute one
                with a mock transport instead of touching the network.
        """
        self.api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
        )
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout = timeout
        self._client = client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion using the Google Gemini API.

        Args:
            prompt: The prompt text.
            **kwargs: ``temperature`` and ``max_tokens`` are forwarded if given.

        Returns:
            The completion text, stripped of leading/trailing whitespace.

        Raises:
            ProviderConfigError: No API key is configured.
            httpx.HTTPStatusError: The API returned a non-2xx response.
            ValueError: The response body was not the expected shape, or the
                prompt was blocked (no candidates returned).
        """
        if not self.api_key:
            raise ProviderConfigError(
                "Gemini API key not configured — set GEMINI_API_KEY or LLM_API_KEY."
            )

        url = self.API_URL_TEMPLATE.format(model=self.model)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.2),
                "maxOutputTokens": kwargs.get("max_tokens", 512),
            },
        }

        data = _post_json(
            self._client,
            url,
            headers={"x-goog-api-key": self.api_key},
            payload=payload,
            timeout=self.timeout,
        )

        candidates = data.get("candidates") or []
        if not candidates:
            # A safety block or an empty generation surfaces here rather than as a
            # KeyError, since it is a distinct, expected condition, not malformed
            # JSON.
            raise ValueError(f"Gemini returned no candidates: {data!r}")

        try:
            parts = candidates[0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected Gemini response shape: {data!r}") from exc


def _post_json(
    client: httpx.Client | None,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST JSON and return the parsed JSON response, raising on any HTTP error.

    Shared by both providers so a caller supplying a mock ``client`` (for tests) or
    swapping providers sees the exact same error-handling behaviour either way.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        response = client.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    finally:
        if owns_client:
            client.close()
