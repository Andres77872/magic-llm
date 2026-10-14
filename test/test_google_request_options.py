"""Offline AI Studio request contract tests through the public MagicLLM facade."""

import asyncio
import json

import pytest

from magic_llm import MagicLLM
from magic_llm.engine import engine_google
from magic_llm.model import ModelChat


@pytest.fixture
def make_engine():
    engines = []

    def create(**options):
        model = options.pop("model", "gemini-3.8-flash")
        engine = MagicLLM(engine="google", model=model, private_key="test-key", retries=1, **options).llm
        engines.append(engine)
        return engine

    yield create
    for engine in engines:
        engine.executor.shutdown(wait=False, cancel_futures=True)


@pytest.fixture
def chat():
    result = ModelChat(system="Be brief.")
    result.add_user_message("Hello")
    return result


@pytest.fixture(params=["sync", "async"])
def prepare(request):
    def build(engine, chat, **options):
        if request.param == "async":
            return asyncio.run(engine.prepare_data(chat, **options))
        return engine.prepare_data_sync(chat, **options)

    return build


def test_catalog_options_produce_native_gemini_payload(make_engine, chat, prepare):
    engine = make_engine(
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        timeout=47,
        temperature=0,
        top_p=0.8,
        max_tokens=128,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        stop="END",
    )
    encoded, headers, data = prepare(engine, chat)

    assert json.loads(encoded) == data
    assert data["generationConfig"] == {
        "maxOutputTokens": 128,
        "stopSequences": ["END"],
    }
    assert data["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert headers["x-goog-api-key"] == "test-key"
    assert "Authorization" not in headers
    assert "test-key" not in encoded.decode()
    assert engine.url_stream == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:streamGenerateContent?alt=sse"
    )


@pytest.mark.parametrize("base_url, version", [
    (None, "v1beta"),
    ("https://generativelanguage.googleapis.com/", "v1beta"),
    ("https://generativelanguage.googleapis.com/v1beta/", "v1beta"),
    ("https://generativelanguage.googleapis.com/v1/models/", "v1"),
])
def test_native_base_url_and_model_resource_names(make_engine, base_url, version):
    engine = make_engine(base_url=base_url, model="models/gemini-3.8-flash")
    root = f"https://generativelanguage.googleapis.com/{version}/models/"
    assert engine.model == "gemini-3.8-flash"
    assert engine.url == root + "gemini-3.8-flash:generateContent"
    assert engine.url_stream == root + "gemini-3.8-flash:streamGenerateContent?alt=sse"
    assert engine.url_tts == root + "gemini-2.5-flash-preview-tts:generateContent"
    assert "base_url" not in engine.kwargs


def test_openai_compatibility_url_requires_openai_engine(make_engine):
    with pytest.raises(ValueError, match="engine='openai'"):
        make_engine(base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def test_call_options_override_defaults_across_aliases(make_engine, chat, prepare):
    engine = make_engine(model="gemini-2.5-flash", topP=0.7, max_tokens=256, stop=["OLD"], temperature=0.5)
    _, _, data = prepare(engine, chat, top_p=0.9, maxOutputTokens=64, stopSequences=None,
                         temperature=0, timeout=19, base_url="https://ignored.example/v1beta")
    assert data["generationConfig"] == {"topP": 0.9, "maxOutputTokens": 64, "temperature": 0}
    assert engine.kwargs["stop"] == ["OLD"]


def test_native_request_fields_stay_outside_generation_config(make_engine, chat, prepare):
    settings = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}]
    engine = make_engine(safety_settings=settings, cached_content="cachedContents/old",
                         thinkingConfig={"thinkingLevel": "LOW"}, responseMimeType="application/json")
    _, _, data = prepare(engine, chat, cachedContent="cachedContents/new", top_k=32)
    assert data["safetySettings"] == settings
    assert data["cachedContent"] == "cachedContents/new"
    assert data["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "LOW"},
        "responseMimeType": "application/json",
    }


@pytest.mark.parametrize("model", [
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "models/gemini-3.8-flash",
    "gemini-3.5-flash-lite", "gemini-3.5-flash-lite-preview", "gemini-3.5-flash",
    "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-4-argon",
])
def test_fixed_sampling_models_omit_ui_defaults_and_native_overrides(make_engine, chat, prepare, model):
    engine = make_engine(model=model, temperature=0.8, top_p=1, top_k=40,
                         presence_penalty=0.5, frequency_penalty=1, max_tokens=128)
    _, _, data = prepare(engine, chat, frequencyPenalty=0.8, presencePenalty=-0.2, topP=0.5,
                         topK=12, temperature=0, thinkingConfig={"thinkingLevel": "LOW"})
    assert data["generationConfig"] == {"maxOutputTokens": 128, "thinkingConfig": {"thinkingLevel": "LOW"}}
    assert engine.kwargs["frequency_penalty"] == 1


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"])
def test_older_models_preserve_sampling_and_penalty_mapping(make_engine, chat, prepare, model):
    engine = make_engine(model=model, temperature=0.8, top_p=0.9, top_k=40,
                         presence_penalty=0.5, frequency_penalty=1, max_tokens=128)
    _, _, data = prepare(engine, chat, frequencyPenalty=0.8)
    assert data["generationConfig"] == {
        "temperature": 0.8, "topP": 0.9, "topK": 40, "presencePenalty": 0.5,
        "frequencyPenalty": 0.8, "maxOutputTokens": 128,
    }


def test_fixed_sampling_omits_even_zero_valued_penalties(make_engine, chat, prepare):
    engine = make_engine(presencePenalty=0, frequency_penalty=0, max_tokens=64)
    _, _, data = prepare(engine, chat)
    assert data["generationConfig"] == {"maxOutputTokens": 64}


@pytest.mark.parametrize("model", ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "models/gemma-4-31b-it"])
def test_hosted_gemma_omits_penalties_and_preserves_sampling(make_engine, chat, prepare, model):
    engine = make_engine(model=model, temperature=0.8, top_p=0.9, top_k=40,
                         presence_penalty=0.5, frequency_penalty=1, max_tokens=128)
    _, _, data = prepare(engine, chat, frequencyPenalty=0.8, presencePenalty=-0.2,
                         temperature=0, stop="END", thinkingConfig={"thinkingLevel": "MINIMAL"})
    assert data["generationConfig"] == {
        "temperature": 0, "topP": 0.9, "topK": 40, "maxOutputTokens": 128,
        "stopSequences": ["END"], "thinkingConfig": {"thinkingLevel": "MINIMAL"},
    }
    assert engine.kwargs["frequency_penalty"] == 1


def test_hosted_gemma_omits_zero_valued_penalties(make_engine, chat, prepare):
    engine = make_engine(model="gemma-4-31b-it", presencePenalty=0, frequency_penalty=0, max_tokens=64)
    _, _, data = prepare(engine, chat, frequencyPenalty=0)
    assert data["generationConfig"] == {"maxOutputTokens": 64}


@pytest.mark.parametrize("operation", ["generate", "async_generate", "stream_generate", "async_stream_generate"])
@pytest.mark.parametrize("call_timeout", [None, 19])
def test_http_receives_timeout_and_native_payload(monkeypatch, make_engine, chat, operation, call_timeout):
    captured = []
    response = {
        "responseId": "test-response",
        "modelVersion": "gemini-3.8-flash",
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    def capture(url, kwargs):
        captured.append((url, kwargs))
        payload = json.loads(kwargs["data"])
        assert payload["generationConfig"] == {"maxOutputTokens": 128}
        assert kwargs["headers"]["x-goog-api-key"] == "test-key"
        assert "test-key" not in url

    class SyncHttp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post_json(self, url, **kwargs):
            capture(url, kwargs)
            return response

        def stream_request(self, method, url, **kwargs):
            assert method == "POST"
            capture(url, kwargs)
            yield ("data: " + json.dumps(response)).encode()

    class AsyncHttp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post_json(self, url, **kwargs):
            capture(url, kwargs)
            return response

        async def post_stream(self, url, **kwargs):
            capture(url, kwargs)
            yield ("data: " + json.dumps(response)).encode()

    monkeypatch.setattr(engine_google, "HttpClient", SyncHttp)
    monkeypatch.setattr(engine_google, "AsyncHttpClient", AsyncHttp)
    engine = make_engine(base_url="https://generativelanguage.googleapis.com/v1beta", timeout=47,
                         top_p=0.8, max_tokens=128)
    options = {} if call_timeout is None else {"timeout": call_timeout}
    if operation == "generate":
        assert engine.generate(chat, **options).content == "Hello"
    elif operation == "async_generate":
        assert asyncio.run(engine.async_generate(chat, **options)).content == "Hello"
    elif operation == "stream_generate":
        assert list(engine.stream_generate(chat, **options))[0].choices[0].delta.content == "Hello"
    else:
        async def collect():
            return [chunk async for chunk in engine.async_stream_generate(chat, **options)]

        assert asyncio.run(collect())[0].choices[0].delta.content == "Hello"
    assert len(captured) == 1
    url, http_options = captured[0]
    assert url == (engine.url_stream if "stream" in operation else engine.url)
    assert http_options["timeout"] == (47 if call_timeout is None else call_timeout)
