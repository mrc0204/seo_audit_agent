"""Tests for the Phase 9 LLM provider implementations (``app/llm/provider.py``).

All three providers accept an injected ``httpx.Client``, so every test here is
served by an ``httpx.MockTransport`` over a fixture response — the same offline
pattern used throughout this project — and never touches the real network or a
real API key.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm.provider import GeminiProvider, GroqProvider, NvidiaProvider, ProviderConfigError

# ---------------------------------------------------------------------------
# GroqProvider
# ---------------------------------------------------------------------------

GROQ_SUCCESS_BODY = {
    "choices": [{"message": {"content": "  Add a unique meta description.  "}}],
}


def _groq_client(status: int = 200, body: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == GroqProvider.API_URL
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(status, json=body if body is not None else GROQ_SUCCESS_BODY)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_groq_generate_returns_stripped_completion_text():
    provider = GroqProvider(api_key="test-key", client=_groq_client())
    assert provider.generate("Suggest a fix") == "Add a unique meta description."


def test_groq_generate_sends_the_prompt_as_a_user_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=GROQ_SUCCESS_BODY)

    provider = GroqProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    provider.generate("What should the fix be?")

    assert captured["body"]["messages"] == [
        {"role": "user", "content": "What should the fix be?"}
    ]


def test_groq_generate_forwards_reasoning_effort_only_when_given():
    # openai/gpt-oss-20b is a reasoning model whose internal "reasoning" pass
    # draws from the same max_tokens budget as its answer -- found live, it can
    # spend the entire budget reasoning and never write an answer.
    # reasoning_effort="low" is qa_agent's fix for that, but it's a Groq/gpt-oss-
    # specific knob: only send it when a caller actually asks for it, so callers
    # on other Groq models (or callers not passing it at all) are unaffected.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=GROQ_SUCCESS_BODY)

    provider = GroqProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    provider.generate("prompt")
    assert "reasoning_effort" not in captured["body"]

    provider.generate("prompt", reasoning_effort="low")
    assert captured["body"]["reasoning_effort"] == "low"


def test_groq_generate_without_an_api_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not make a network call with no API key")

    provider = GroqProvider(
        api_key=None, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ProviderConfigError):
        provider.generate("anything")


def test_groq_generate_raises_on_http_error_status():
    provider = GroqProvider(api_key="test-key", client=_groq_client(status=429, body={"error": "rate limited"}))
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate("anything")


def test_groq_generate_raises_a_clear_error_on_unexpected_response_shape():
    provider = GroqProvider(api_key="test-key", client=_groq_client(body={"unexpected": "shape"}))
    with pytest.raises(ValueError):
        provider.generate("anything")


def test_groq_model_defaults_and_is_overridable():
    from app.llm.provider import DEFAULT_GROQ_MODEL

    assert GroqProvider(api_key="k").model == DEFAULT_GROQ_MODEL
    assert GroqProvider(api_key="k", model="llama-other").model == "llama-other"


def test_groq_falls_back_to_llm_api_key_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    assert GroqProvider().api_key == "from-env"


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

GEMINI_SUCCESS_BODY = {
    "candidates": [{"content": {"parts": [{"text": " Write a unique title. "}]}}],
}


def _gemini_client(status: int = 200, body: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(status, json=body if body is not None else GEMINI_SUCCESS_BODY)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_generate_returns_stripped_completion_text():
    provider = GeminiProvider(api_key="test-key", client=_gemini_client())
    assert provider.generate("Suggest a fix") == "Write a unique title."


def test_gemini_generate_joins_multiple_parts():
    body = {"candidates": [{"content": {"parts": [{"text": "Part one. "}, {"text": "Part two."}]}}]}
    provider = GeminiProvider(api_key="test-key", client=_gemini_client(body=body))
    assert provider.generate("q") == "Part one. Part two."


def test_gemini_generate_without_an_api_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not make a network call with no API key")

    provider = GeminiProvider(
        api_key=None, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ProviderConfigError):
        provider.generate("anything")


def test_gemini_generate_raises_on_http_error_status():
    provider = GeminiProvider(api_key="test-key", client=_gemini_client(status=400, body={"error": "bad request"}))
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate("anything")


def test_gemini_generate_raises_when_no_candidates_returned():
    # A safety block or empty generation is a distinct, expected condition, not
    # malformed JSON — it must surface as a clear ValueError either way.
    provider = GeminiProvider(api_key="test-key", client=_gemini_client(body={"candidates": []}))
    with pytest.raises(ValueError):
        provider.generate("anything")


def test_gemini_model_is_used_in_the_request_url():
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=GEMINI_SUCCESS_BODY)

    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-custom",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.generate("q")
    assert "gemini-custom" in seen_urls[0]


def test_gemini_falls_back_to_llm_api_key_env_var(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    assert GeminiProvider().api_key == "from-env"


# ---------------------------------------------------------------------------
# NvidiaProvider -- added as a third option specifically because Groq's and
# Gemini's free tiers each hit a real ceiling during this project's own live
# testing; NVIDIA's OpenAI-compatible NIM catalog draws from a separate quota.
# ---------------------------------------------------------------------------

NVIDIA_SUCCESS_BODY = {
    "choices": [{"message": {"content": "  Add a unique meta description.  "}}],
}


def _nvidia_client(status: int = 200, body: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == NvidiaProvider.API_URL
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(status, json=body if body is not None else NVIDIA_SUCCESS_BODY)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_nvidia_generate_returns_stripped_completion_text():
    provider = NvidiaProvider(api_key="test-key", client=_nvidia_client())
    assert provider.generate("Suggest a fix") == "Add a unique meta description."


def test_nvidia_generate_sends_the_prompt_as_a_user_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=NVIDIA_SUCCESS_BODY)

    provider = NvidiaProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    provider.generate("What should the fix be?")

    assert captured["body"]["messages"] == [
        {"role": "user", "content": "What should the fix be?"}
    ]


def test_nvidia_generate_ignores_provider_specific_kwargs_it_does_not_recognize():
    # Groq's reasoning_effort must not break a call routed to NVIDIA when a
    # caller passes through the same kwarg set used for every provider.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=NVIDIA_SUCCESS_BODY)

    provider = NvidiaProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    provider.generate("prompt", reasoning_effort="low")
    assert "reasoning_effort" not in captured["body"]


def test_nvidia_thinking_mode_is_explicitly_disabled_by_default():
    # Found live: DEFAULT_NVIDIA_MODEL writes its reasoning straight into the
    # answer's own content by default, and merely OMITTING chat_template_kwargs
    # does NOT turn thinking off for this model -- a "say hello" prompt came
    # back as an unfinished "Here's a thinking process: ..." even at
    # max_tokens=300. Sending chat_template_kwargs.enable_thinking=False
    # explicitly is what a live call confirmed actually produces a clean
    # "Hello!" answer, so it must be sent every call, not just implied by
    # absence.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=NVIDIA_SUCCESS_BODY)

    provider = NvidiaProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    provider.generate("prompt")
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_budget" not in captured["body"]


def test_nvidia_thinking_mode_can_be_explicitly_opted_into():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=NVIDIA_SUCCESS_BODY)

    provider = NvidiaProvider(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    provider.generate("prompt", enable_thinking=True, reasoning_budget=16384)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["body"]["reasoning_budget"] == 16384


def test_nvidia_generate_without_an_api_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not make a network call with no API key")

    provider = NvidiaProvider(
        api_key=None, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ProviderConfigError):
        provider.generate("anything")


def test_nvidia_generate_raises_on_http_error_status():
    provider = NvidiaProvider(
        api_key="test-key", client=_nvidia_client(status=429, body={"error": "rate limited"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate("anything")


def test_nvidia_generate_raises_a_clear_error_on_unexpected_response_shape():
    provider = NvidiaProvider(api_key="test-key", client=_nvidia_client(body={"unexpected": "shape"}))
    with pytest.raises(ValueError):
        provider.generate("anything")


def test_nvidia_model_defaults_and_is_overridable():
    from app.llm.provider import DEFAULT_NVIDIA_MODEL

    assert NvidiaProvider(api_key="k").model == DEFAULT_NVIDIA_MODEL
    assert NvidiaProvider(api_key="k", model="meta/other-model").model == "meta/other-model"


def test_nvidia_falls_back_to_llm_api_key_env_var(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    assert NvidiaProvider().api_key == "from-env"


def test_build_llm_generate_wires_up_nvidia_by_name():
    from app.main import build_llm_generate

    generate = build_llm_generate("nvidia", api_key="test-key")
    assert generate is not None
    assert generate.__self__.__class__ is NvidiaProvider


# ---------------------------------------------------------------------------
# All three providers plug into the existing structural LLM boundaries unchanged
# ---------------------------------------------------------------------------


def test_groq_provider_plugs_into_the_seo_agent_llm_boundary():
    from app.agents.seo_agent import apply_llm_suggested_fixes
    from app.models.finding import Finding

    provider = GroqProvider(api_key="test-key", client=_groq_client())
    finding = Finding(
        metric="missing_meta_description",
        page="https://x.example/",
        severity="warning",
        evidence="No meta description present.",
        suggested_fix="deterministic text",
        check_id="SEO-META-001",
    )
    updated = apply_llm_suggested_fixes([finding], provider.generate)[0]

    assert updated.suggested_fix == "Add a unique meta description."
    # Structural guarantee from Phase 4 still holds regardless of which provider
    # is plugged in.
    assert updated.evidence == finding.evidence
    assert updated.check_id == finding.check_id


def test_a_provider_exception_still_degrades_to_null_in_qa_agent(monkeypatch):
    from app.agents.qa_agent import answer_question
    from app.extraction.seo_extractor import build_page_data

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider = GroqProvider(api_key=None)  # raises ProviderConfigError, no network
    page = build_page_data(
        "https://x.example/",
        "https://x.example/",
        200,
        "<html><body><main><p>Some real page content lives right here today.</p></main></body></html>",
    )
    answer = answer_question("What content is here?", [page], generate=provider.generate)
    # qa_agent fails CLOSED: a raising provider must not produce a guessed answer.
    assert answer.url is None
    assert answer.match_type == "none"
