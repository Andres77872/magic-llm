import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat

# Provider configurations: (provider_name, key_name_in_json, success_model, fail_model)
TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o", "gpt-4o1"),
    # ("lepton", "lepton", "llama3-1-8b", "FAIL/llama3-1-8b"),
    ("amazon", "amazon", "amazon.nova-pro-v1:0", "amazon.titan-text-lite-v3"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307"),
    ("cloudflare", "cloudflare", "@cf/meta/llama-2-7b-chat-int8", "FAIL/@cf/meta/llama-2-7b-chat-int8"),
    ("cohere", "cohere", "command-light", "FAIL/command-light"),
    ("google", "google", "gemini-2.5-flash", "FAIL/gemini-1.5-flash"),
    ("Cerebras", "Cerebras", "llama3.1-8b", "FAIL/llama3.1-8b"),
    ("SambaNova", "SambaNova", "Meta-Llama-3.1-8B-Instruct", "FAIL/Meta-Llama-3.1-8B-Instruct"),
    ("deepinfra", "deepinfra", "microsoft/WizardLM-2-8x22B", "microsoft/WizardLM-2-8x22B-model-fail"),
    ("deepseek", "deepseek", "deepseek-chat", "FAIL/deepseek-chat"),
    ("parasail", "parasail", "parasail-mistral-nemo", "parasail-mistral-nemo-fail"),
    ("x.ai", "x.ai", "grok-3-mini", "grok-3-mini-fail"),
    ("together.ai", "together.ai", "meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-8b-chat-hf-fail"),
    ("perplexity", "perplexity", "sonar", "sonar-fail"),
    ("openrouter", "openrouter", "mistralai/mistral-nemo", "mistralai/mistral-nemo-fail"),
    ("novita.ai", "novita.ai", "mistralai/mistral-nemo", "FAIL/mistralai/mistral-nemo"),
    ("mistral", "mistral", "open-mistral-7b", "FAIL/open-mistral-7b"),
    ("hyperbolic", "hyperbolic", "meta-llama/Meta-Llama-3.1-8B-Instruct", "FAIL/meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("groq", "groq", "llama3-8b-8192", "FAIL/llama3-8b-8192"),
    ("fireworks.ai", "fireworks.ai", "accounts/fireworks/models/llama4-scout-instruct-basic", "accounts/fireworks/models/llama4-scout-instruct-basic-fail"),
]

# Locate keys file via environment variable or default to test/keys.json
KEYS_FILE = os.getenv(
    "MAGIC_LLM_KEYS",
    "/home/andres/Documents/keys.json",
)
if not os.path.exists(KEYS_FILE):
    pytest.skip(
        f"No keys file found at {KEYS_FILE}. "
        "Set MAGIC_LLM_KEYS env var or place keys.json in this directory.",
        allow_module_level=True,
    )
with open(KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

# Filter providers for which keys are provided
PROVIDERS = [
    (provider, key_name, success_model, fail_model)
    for provider, key_name, success_model, fail_model in TEST_PROVIDERS
    if key_name in ALL_KEYS
]
if not PROVIDERS:
    pytest.skip("No matching providers found in keys file", allow_module_level=True)


def _build_chat():
    """Construct a simple chat with a single user message."""
    chat = ModelChat()
    chat.add_user_message("Hello")
    return chat


# Streaming tests (sync & async)
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_stream_generate(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, **keys)
    content = ""
    for chunk in client.llm.stream_generate(chat):
        content += chunk.choices[0].delta.content or ""
    assert content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_stream_generate(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, **keys)
    content = ""
    async for chunk in client.llm.async_stream_generate(chat):
        content += chunk.choices[0].delta.content or ""
    assert content


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_stream_generate_fail(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=fail_model, **keys)
    with pytest.raises(ChatException):
        for chunk in client.llm.stream_generate(chat):
            _ = chunk.choices[0].delta.content or ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_stream_generate_fail(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=fail_model, **keys)
    with pytest.raises(ChatException):
        async for chunk in client.llm.async_stream_generate(chat):
            _ = chunk.choices[0].delta.content or ""


# Non-streaming tests
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_non_stream_generate(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # valid model → succeeds
    good = MagicLLM(model=model, **keys)
    resp = good.llm.generate(chat)
    assert resp.content, "Expected non‐empty content"

    # invalid model → ChatException
    bad = MagicLLM(model=fail_model, **keys)
    with pytest.raises(ChatException):
        bad.llm.generate(chat)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_non_stream_generate(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    good = MagicLLM(model=model, **keys)
    resp = await good.llm.async_generate(chat)
    assert resp.content, "Expected non‐empty content"

    bad = MagicLLM(model=fail_model, **keys)
    with pytest.raises(ChatException):
        await bad.llm.async_generate(chat)


# Fallback tests
def _make_fallback_client(key_name, success_model):
    keys = dict(ALL_KEYS[key_name])
    return MagicLLM(model=success_model, **keys)


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_non_stream_fallback(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(
        model=fail_model,
        fallback=_make_fallback_client(key_name, model),
        **keys,
    )
    resp = client.llm.generate(chat)
    assert resp.content, "Expected fallback content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_non_stream_fallback(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(
        model=fail_model,
        fallback=_make_fallback_client(key_name, model),
        **keys,
    )
    resp = await client.llm.async_generate(chat)
    assert resp.content, "Expected fallback content"


# Usage & callback tests (streaming)
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_generate_usage_and_callback(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    calls = []

    def cb(msg: ModelChat, content: str, usage, model_name: str, meta):
        calls.append((content, usage, model_name))

    client = MagicLLM(
        model=fail_model,
        fallback=_make_fallback_client(key_name, model),
        callback=cb,
        **keys,
    )
    output = ""
    for chunk in client.llm.stream_generate(chat):
        output += chunk.choices[0].delta.content or ""

    assert calls, "Callback was not invoked"
    last_usage = calls[-1][1]
    assert last_usage.prompt_tokens > 0
    assert last_usage.completion_tokens > 0
    assert output, "Expected some streamed content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_generate_usage_and_callback(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    calls = []

    def cb(msg: ModelChat, content: str, usage, model_name: str, meta):
        calls.append((content, usage, model_name))

    client = MagicLLM(
        model=fail_model,
        fallback=_make_fallback_client(key_name, model),
        callback=cb,
        **keys,
    )
    output = ""
    async for chunk in client.llm.async_stream_generate(chat):
        output += chunk.choices[0].delta.content or ""

    assert calls, "Callback was not invoked"
    last_usage = calls[-1][1]
    assert last_usage.prompt_tokens > 0
    assert last_usage.completion_tokens > 0
    assert output, "Expected some streamed content"


# Usage tests (non-streaming)
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_non_stream_usage(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, **keys)
    resp = client.llm.generate(chat)
    u = resp.usage
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0
    # in non‐streaming, no first-token latency recorded
    assert u.ttft == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
async def test_async_non_stream_usage(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, **keys)
    resp = await client.llm.async_generate(chat)
    u = resp.usage
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0
    assert u.ttft == 0