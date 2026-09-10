"""Swappable LLM provider interface protocol and client implementations.

All three providers talk to their vendor's plain REST endpoint over ``httpx``
rather than a vendor SDK. This is a deliberate simplification, not a missing
feature: each API is one HTTP POST and a JSON body, ``httpx`` is already a
project dependency, and skipping the SDKs means selecting a provider never
requires an extra install. All three are the vendors' documented **free-tier**
endpoints — no paid subscription, matching the brief's constraint. NVIDIA was
added as a third option specifically because Groq's and Gemini's free tiers each
hit a real ceiling during this project's own live testing (Groq's daily token
cap, in particular) — it draws from an entirely separate quota, so switching
``LLM_PROVIDER`` is a genuine way to keep testing rather than waiting out a reset.

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
import time
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

DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


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


class NvidiaProvider(LLMProvider):
    """NVIDIA NIM LLM provider implementation.

    Talks to NVIDIA's OpenAI-compatible NIM catalog endpoint
    (``integrate.api.nvidia.com``) — free API keys (``nvapi-...``) are issued at
    build.nvidia.com. Added as a third option alongside Groq/Gemini specifically
    because both of those hit real free-tier ceilings during this project's own
    live testing (Groq's daily token cap, in particular) — NVIDIA's free tier is a
    separate quota entirely, so it is a genuine fallback, not a duplicate of an
    existing one. Still a free tier with its own real limits (a fixed signup
    credit balance and its own requests-per-minute cap), not unlimited.
    """

    API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            api_key: NVIDIA NIM API key. Falls back to the ``NVIDIA_API_KEY`` env
                var, then ``LLM_API_KEY`` (the name used in ``.env.example`` when
                ``LLM_PROVIDER=nvidia``).
            model: Model id. Falls back to ``NVIDIA_MODEL``, then
                :data:`DEFAULT_NVIDIA_MODEL`.
            timeout: Per-request timeout in seconds.
            client: Optional pre-built ``httpx.Client`` — tests substitute one
                with a mock transport instead of touching the network.
        """
        self.api_key = (
            api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY")
        )
        self.model = model or os.environ.get("NVIDIA_MODEL") or DEFAULT_NVIDIA_MODEL
        self.timeout = timeout
        self._client = client

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text completion using the NVIDIA NIM API.

        Args:
            prompt: The prompt text.
            **kwargs: ``temperature`` and ``max_tokens`` are forwarded if given.
                ``enable_thinking``, if explicitly given as ``True``, turns
                thinking mode back on (see the note below for why it is
                explicitly turned OFF otherwise, rather than merely left
                unset), and ``reasoning_budget`` is forwarded alongside it.
                Anything else (e.g. Groq-specific ``reasoning_effort``) is
                ignored, so callers can pass through the same keyword set used
                elsewhere without this provider breaking. Always a single
                non-streamed response — this provider's interface is a plain
                ``str`` return, and every call site in this codebase already
                expects that, not a token stream.

        Returns:
            The completion text, stripped of leading/trailing whitespace.

        Raises:
            ProviderConfigError: No API key is configured.
            httpx.HTTPStatusError: The API returned a non-2xx response.
            ValueError: The response body was not the expected shape.

        Note on ``enable_thinking``/``reasoning_budget``:
            Nemotron models on NVIDIA's catalog (this was confirmed live on
            ``nvidia/nemotron-3.5-lightning-30b-a3b`` before
            :data:`DEFAULT_NVIDIA_MODEL` was changed to a larger Nemotron variant
            — not yet re-confirmed on the new one) are "thinking" models whose
            reasoning, by default, is written straight into the answer's own
            ``content`` field rather than a separate field the way Groq's
            reasoning model does — found live: with nothing sent for
            ``chat_template_kwargs`` at all, a plain "say hello" prompt came back
            as "Here's a thinking process: 1. Analyze User Request..." and never
            finished reasoning to an actual answer even at ``max_tokens=300``.
            Merely omitting the field does NOT default to thinking-off for this
            model family, unlike what its own parameter name might suggest —
            confirmed live that ``chat_template_kwargs.enable_thinking`` must be
            sent explicitly as ``False`` to get a clean, parseable answer
            (confirmed live: the identical prompt then returned exactly
            ``"Hello!"`` with
            ``reasoning_content: null``). So it is sent explicitly every call,
            defaulting to ``False``, and only flipped to ``True`` when a caller
            opts in.
        """
        if not self.api_key:
            raise ProviderConfigError(
                "NVIDIA API key not configured — set NVIDIA_API_KEY or LLM_API_KEY."
            )

        enable_thinking = bool(kwargs.get("enable_thinking", False))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 512),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if enable_thinking and kwargs.get("reasoning_budget"):
            payload["reasoning_budget"] = kwargs["reasoning_budget"]

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
            raise ValueError(f"Unexpected NVIDIA response shape: {data!r}") from exc


def _post_json(
    client: httpx.Client | None,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    max_retries: int = 3,
) -> dict[str, Any]:
    """POST JSON and return the parsed JSON response, raising on any HTTP error.

    Shared by both providers so a caller supplying a mock ``client`` (for tests) or
    swapping providers sees the exact same error-handling behaviour either way.
    Includes exponential-backoff retries for transient 429/5xx status codes and network drops.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    last_exc: Exception | None = None
    try:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(url, headers=headers, json=payload, timeout=timeout)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("POST JSON failed after retries")
    finally:
        if owns_client:
            client.close()
