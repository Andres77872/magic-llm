"""Discovery hardening: adapter resolution, URLs, errors, pagination, metadata.

Regression coverage for defects found validating discovery against live
provider APIs (payload shapes below are trimmed copies of real listings).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.discovery import (
    DiscoveryAuthError,
    DiscoveryError,
    resolve_discovery_engine,
)
from magic_llm.engine.discovery.anthropic_discovery import AnthropicDiscoveryAdapter
from magic_llm.engine.discovery.base_discovery import versioned_endpoint
from magic_llm.engine.discovery.capabilities import DeclaredFieldStrategy
from magic_llm.engine.discovery.cohere_discovery import CohereDiscoveryAdapter
from magic_llm.engine.discovery.google_discovery import GoogleDiscoveryAdapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
    extract_listing_pricing,
)
from magic_llm.engine.discovery.openai_compatible.perplexity import PerplexityDiscoveryAdapter
from magic_llm.engine.discovery.openai_discovery import OpenAIDiscoveryAdapter
from magic_llm.engine.discovery.openrouter_discovery import OpenRouterDiscoveryAdapter
from magic_llm.engine.discovery.sambanova_discovery import SambaNovaDiscoveryAdapter
from magic_llm.util.http import AsyncHttpClient, HttpError


# ── Adapter resolution ────────────────────────────────────────────────────


@pytest.mark.parametrize("base_url,expected", [
    ("https://api.deepinfra.com/v1/openai", "deepinfra"),
    ("https://api.groq.com/openai/v1", "groq"),
    ("https://openrouter.ai/api/v1", "openrouter"),
    ("https://api.together.xyz/v1", "together"),
    ("https://api.mistral.ai/v1", "mistral"),
    ("https://api.x.ai/v1", "xai"),
    ("https://api.studio.nebius.com/v1", "nebius"),
    ("https://api.novita.ai/v3/openai", "novita"),
    ("https://api.perplexity.ai", "perplexity"),
    ("https://api.sambanova.ai/v1", "sambanova"),
    ("https://api.openai.com/v1", "openai"),
    ("https://api.fireworks.ai/inference/v1", "openai"),  # no dedicated adapter
    (None, "openai"),
])
def test_openai_engine_resolves_by_host(base_url, expected):
    assert resolve_discovery_engine("openai", base_url) == expected


def test_resolution_keeps_native_engines_and_rejects_unknown():
    assert resolve_discovery_engine("Anthropic", "https://api.anthropic.com") == "anthropic"
    assert resolve_discovery_engine("amazon") is None
    assert resolve_discovery_engine("cloudflare") is None


def test_host_match_requires_label_boundary():
    assert resolve_discovery_engine("openai", "https://evilapi.groq.com.example/v1") == "openai"
    assert resolve_discovery_engine("openai", "https://eu.api.groq.com/v1") == "groq"


# ── Listing URLs never duplicate the version ──────────────────────────────


@pytest.mark.parametrize("base,expected", [
    ("https://api.sambanova.ai", "https://api.sambanova.ai/v1/models"),
    ("https://api.sambanova.ai/v1", "https://api.sambanova.ai/v1/models"),
    ("https://api.sambanova.ai/v1/", "https://api.sambanova.ai/v1/models"),
    ("https://x.test/v1/models", "https://x.test/v1/models"),
])
def test_versioned_endpoint(base, expected):
    assert versioned_endpoint(base, "v1") == expected


@pytest.mark.parametrize("adapter,expected", [
    (SambaNovaDiscoveryAdapter(base_url="https://api.sambanova.ai/v1"), "https://api.sambanova.ai/v1/models"),
    (OpenRouterDiscoveryAdapter(base_url="https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1/models"),
    (AnthropicDiscoveryAdapter(base_url="https://api.anthropic.com/v1"), "https://api.anthropic.com/v1/models"),
    (CohereDiscoveryAdapter(base_url="https://api.cohere.com/v2"), "https://api.cohere.com/v1/models"),
    (GoogleDiscoveryAdapter(base_url="https://generativelanguage.googleapis.com/v1beta"),
     "https://generativelanguage.googleapis.com/v1beta/models"),
    (PerplexityDiscoveryAdapter(base_url="https://api.perplexity.ai"), "https://api.perplexity.ai/v1/models"),
    (PerplexityDiscoveryAdapter(), "https://api.perplexity.ai/v1/models"),
])
def test_adapters_accept_chat_base_urls(adapter, expected):
    assert adapter._get_endpoint_url() == expected


# ── Error classification ──────────────────────────────────────────────────


def _raise_sync(adapter, error):
    with patch("magic_llm.engine.discovery.base_discovery.HttpClient") as mock:
        inst = MagicMock()
        inst.request.side_effect = error
        inst.__enter__.return_value = inst
        mock.return_value = inst
        return adapter.discover()


def test_403_is_auth_error():
    with pytest.raises(DiscoveryAuthError):
        _raise_sync(OpenAIDiscoveryAdapter(api_key="k"), HttpError("x", status_code=403))


def test_google_invalid_key_400_is_auth_error():
    body = json.dumps({"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                 "status": "INVALID_ARGUMENT"}}).encode()
    with pytest.raises(DiscoveryAuthError):
        _raise_sync(GoogleDiscoveryAdapter(api_key="k"), HttpError("x", status_code=400, response_content=body))


def test_plain_400_stays_generic_with_body_preview():
    with pytest.raises(DiscoveryError) as exc:
        _raise_sync(OpenAIDiscoveryAdapter(api_key="k"),
                    HttpError("x", status_code=400, response_content=b'{"error": "bad request shape"}'))
    assert not isinstance(exc.value, DiscoveryAuthError)
    assert "bad request shape" in str(exc.value)


def test_transport_error_without_status_does_not_crash():
    """HttpError(status_code=None) used to raise TypeError in the 5xx comparison."""
    with pytest.raises(DiscoveryError) as exc:
        _raise_sync(OpenAIDiscoveryAdapter(api_key="k", base_url="https://down.test/v1"),
                    HttpError("requests error: connection refused"))
    assert "unreachable" in str(exc.value)
    assert "https://down.test/v1/models" in str(exc.value)


@pytest.mark.asyncio
async def test_async_client_normalizes_total_timeout():
    client = AsyncHttpClient()
    client.session = MagicMock()
    client.session.request = MagicMock(side_effect=asyncio.TimeoutError())
    with pytest.raises(HttpError) as exc:
        await client.request("GET", "https://slow.test", timeout=1)
    assert exc.value.status_code is None
    assert "timed out" in str(exc.value)


# ── Pagination ────────────────────────────────────────────────────────────


def _paged_client(pages):
    patcher = patch("magic_llm.engine.discovery.base_discovery.HttpClient")
    mock = patcher.start()
    inst = MagicMock()
    inst.request.side_effect = [json.dumps(p).encode() for p in pages]
    inst.__enter__.return_value = inst
    mock.return_value = inst
    return patcher, inst


def test_google_follows_next_page_token():
    patcher, inst = _paged_client([
        {"models": [{"name": "models/gemini-2.5-pro"}], "nextPageToken": "tok/1"},
        {"models": [{"name": "models/gemini-3-flash"}]},
    ])
    try:
        models = GoogleDiscoveryAdapter(api_key="k").discover()
    finally:
        patcher.stop()
    assert [m.external_id for m in models] == ["gemini-2.5-pro", "gemini-3-flash"]
    assert inst.request.call_args_list[1].args[1].endswith("pageSize=1000&pageToken=tok%2F1")
    assert all(m.capabilities.vision for m in models)  # Gemini 2.x and 3.x accept images


def test_cohere_follows_next_page_token():
    patcher, inst = _paged_client([
        {"models": [{"name": "command-a"}], "next_page_token": "abc"},
        {"models": [{"name": "embed-v4.0"}], "next_page_token": ""},
    ])
    try:
        models = CohereDiscoveryAdapter(api_key="k").discover()
    finally:
        patcher.stop()
    assert [m.external_id for m in models] == ["command-a", "embed-v4.0"]
    assert inst.request.call_count == 2


def test_anthropic_follows_has_more():
    patcher, inst = _paged_client([
        {"data": [{"id": "claude-opus-4-5"}], "has_more": True, "last_id": "claude-opus-4-5"},
        {"data": [{"id": "claude-haiku-4-5"}], "has_more": False},
    ])
    try:
        models = AnthropicDiscoveryAdapter(api_key="k").discover()
    finally:
        patcher.stop()
    assert [m.external_id for m in models] == ["claude-opus-4-5", "claude-haiku-4-5"]
    assert "after_id=claude-opus-4-5" in inst.request.call_args_list[1].args[1]
    assert all(m.context_window == 200000 for m in models)


def test_extra_headers_sent_under_adapter_auth():
    patcher, inst = _paged_client([{"data": []}])
    try:
        OpenAIDiscoveryAdapter(api_key="k", extra_headers={"OpenAI-Organization": "org", "Authorization": "x"}).discover()
    finally:
        patcher.stop()
    headers = inst.request.call_args.kwargs["headers"]
    assert headers["OpenAI-Organization"] == "org"
    assert headers["Authorization"] == "Bearer k"


# ── Pricing extraction (per 1M tokens, USD) ───────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    # OpenRouter / Groq / SambaNova: per-token strings
    ({"pricing": {"prompt": "0.00000015", "completion": "0.0000006"}}, (0.15, 0.6)),
    # OpenRouter router sentinel
    ({"pricing": {"prompt": "-1", "completion": "-1"}}, None),
    # Novita decimal per-M
    ({"pricing": {"prompt": {"price_per_m": 1500, "price_per_m_decimal": "0.15"},
                  "completion": {"price_per_m": 5000, "price_per_m_decimal": "0.5"}}}, (0.15, 0.5)),
    # Together: already per 1M
    ({"pricing": {"hourly": 0, "input": 1.04, "output": 1.04}}, (1.04, 1.04)),
    # DeepInfra metadata
    ({"metadata": {"pricing": {"input_tokens": 0.1, "output_tokens": 0.15}}}, (0.1, 0.15)),
    # Hyperbolic
    ({"input_price": 1.25, "output_price": 1.25}, (1.25, 1.25)),
    # xAI: USD cents per 100M tokens
    ({"prompt_text_token_price": 12500, "completion_text_token_price": 25000}, (1.25, 2.5)),
    ({}, None),
])
def test_extract_listing_pricing(raw, expected):
    pricing = extract_listing_pricing(raw)
    if expected is None:
        assert pricing is None
    else:
        assert (pricing.input_per_million, pricing.output_per_million) == expected


def test_openrouter_router_has_no_negative_price():
    adapter = OpenRouterDiscoveryAdapter()
    [model] = adapter._normalize_response({"data": [{
        "id": "openrouter/auto", "context_length": 2000000,
        "pricing": {"prompt": "-1", "completion": "-1"},
        "architecture": {"modality": "text+image->text+image",
                         "input_modalities": ["text", "image"], "output_modalities": ["text", "image"]},
        "supported_parameters": ["tools", "include_reasoning"],
    }]})
    assert model.pricing is None
    assert model.capabilities.vision is True          # current architecture shape
    assert model.capabilities.image_output is True
    assert model.capabilities.function_calling is True
    assert model.capabilities.reasoning is True


# ── Declared capability fields & token limits ─────────────────────────────


def test_mistral_capabilities_and_context():
    [model] = OpenAIDiscoveryAdapter(api_key="k")._normalize_response({"data": [{
        "id": "mistral-large-latest", "max_context_length": 262144,
        "capabilities": {"completion_chat": True, "function_calling": True, "vision": True,
                         "reasoning": False, "audio_transcription": False, "audio_speech": False},
    }]})
    assert model.context_window == 262144
    assert model.capabilities.vision is True
    assert model.capabilities.function_calling is True


def test_together_embedding_type_is_not_chat_or_tools():
    [model] = OpenAIDiscoveryAdapter(api_key="k")._normalize_response([
        {"id": "BAAI/bge-large-en-v1.5", "type": "embedding", "context_length": 512},
    ])
    caps = model.capabilities
    assert caps.embedding is True
    assert caps.chat is False
    assert caps.function_calling is False
    assert caps.streaming is False


def test_groq_whisper_is_transcription_not_chat():
    [model] = OpenAIDiscoveryAdapter(api_key="k")._normalize_response({"data": [{
        "id": "whisper-large-v3", "input_modalities": ["audio"], "output_modalities": ["text"],
        "supported_features": [],
    }]})
    assert model.capabilities.audio_input is True
    assert model.capabilities.chat is False
    assert model.capabilities.function_calling is False


def test_declared_fields_absent_yields_nothing():
    assert DeclaredFieldStrategy().infer("openai", "gpt-x", {"id": "gpt-x", "object": "model"}) == {}


@pytest.mark.parametrize("raw,expected_ctx,expected_out", [
    ({"id": "a", "context_size": 1048576, "max_output_tokens": 131072}, 1048576, 131072),           # Novita
    ({"id": "b", "metadata": {"context_length": 262144}}, 262144, None),                            # DeepInfra
    ({"id": "c", "context_window": 131072, "max_output_length": 16384}, 131072, 16384),             # Groq
    ({"id": "d", "top_provider": {"context_length": 64000, "max_completion_tokens": 8000}}, 64000, 8000),
])
def test_openai_compatible_token_aliases(raw, expected_ctx, expected_out):
    [model] = OpenAIDiscoveryAdapter(api_key="k")._normalize_response({"data": [raw]})
    assert model.context_window == expected_ctx
    assert model.max_output_tokens == expected_out


@pytest.mark.parametrize("model_id,expected", [
    ("gpt-4o-2024-08-06", 128000),
    ("gpt-4o-mini", 128000),
    ("gpt-4-0613", 8192),
    ("gpt-4.1-mini", 1047576),
    ("gpt-4-turbo", 128000),
    ("o3-mini", 200000),
])
def test_context_window_map_prefers_specific_families(model_id, expected):
    """``gpt-4`` used to match first and give every GPT-4o/4.1 model 8K."""
    assert OpenAICompatibleAdapter._context_window_map_fallback({"id": model_id}) == expected


def test_cohere_current_feature_names():
    [chat, vision, stt] = CohereDiscoveryAdapter(api_key="k")._normalize_response({"models": [
        {"name": "command-a-03-2025", "endpoints": ["chat"], "context_length": 288000,
         "features": ["json_mode", "strict_tools", "tools", "tool_choice", "citations"]},
        {"name": "c4ai-aya-vision-32b", "endpoints": ["chat"], "features": ["logprobs", "vision"]},
        {"name": "cohere-transcribe-03-2026", "endpoints": ["transcriptions"], "features": None},
    ]})
    assert chat.capabilities.function_calling is True
    assert vision.capabilities.vision is True
    assert vision.capabilities.function_calling is False
    assert stt.capabilities.audio_input is True
    assert stt.capabilities.chat is False


def test_groq_speech_output_is_audio_output():
    [model] = OpenAIDiscoveryAdapter(api_key="k")._normalize_response({"data": [{
        "id": "canopylabs/orpheus-v1-english", "input_modalities": ["text"], "output_modalities": ["speech"],
    }]})
    assert model.capabilities.audio_output is True
    assert model.capabilities.chat is False
