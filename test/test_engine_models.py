import os
import pytest
import json
from magic_llm import MagicLLM
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat

# Provider configurations: (provider_name, key_name_in_json, success_model, fail_model)
TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o", "gpt-4o1"),
    ("lepton", "lepton", "llama3-1-8b", "FAIL/llama3-1-8b"),
    ("amazon", "amazon", "amazon.nova-pro-v1:0", "amazon.titan-text-lite-v3"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307"),
    ("cloudflare", "cloudflare", "@cf/meta/llama-2-7b-chat-int8", "FAIL/@cf/meta/llama-2-7b-chat-int8"),
    ("cohere", "cohere", "command-light", "FAIL/command-light"),
    ("google", "google", "gemini-1.5-flash", "FAIL/gemini-1.5-flash"),
    ("Cerebras", "Cerebras", "llama3.1-8b", "FAIL/llama3.1-8b"),
    ("SambaNova", "SambaNova", "Meta-Llama-3.1-8B-Instruct", "FAIL/Meta-Llama-3.1-8B-Instruct"),
    ("deepinfra", "deepinfra", "microsoft/WizardLM-2-8x22B", "microsoft/WizardLM-2-8x22B-model-fail"),
    ("deepseek", "deepseek", "deepseek-chat", "FAIL/deepseek-chat"),
]

# Locate keys file via environment variable or default to test/keys.json
KEYS_FILE = os.getenv(
    "MAGIC_LLM_KEYS",
    os.path.join(os.path.dirname(__file__), "keys.json"),
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

@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_sync_stream_generate(provider, key_name, model, fail_model):
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, **keys)
    content = ''
    for i in client.llm.stream_generate(chat):
        content += i.choices[0].delta.content or ''
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
    content = ''
    async for i in client.llm.astream_generate(chat):
        content += i.choices[0].delta.content or ''
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
        content = ''
        for i in client.llm.stream_generate(chat):
            content += i.choices[0].delta.content or ''

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
        content = ''
        async for i in client.llm.astream_generate(chat):
            content += i.choices[0].delta.content or ''
