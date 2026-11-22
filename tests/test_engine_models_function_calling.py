from magic_llm import MagicLLM

import pytest

from magic_llm.model import ModelChat, ModelChatResponse

from conftest import assert_forced_weather_response, assert_forced_weather_stream, get_provider_key

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

FUNCTION_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o", "gpt-4o1"),
    # ("lepton", "lepton", "llama3-1-8b", "FAIL/llama3-1-8b"),
    # ("amazon", "amazon", "amazon.nova-pro-v1:0", "amazon.titan-text-lite-v3"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307"),
    # ("cloudflare", "cloudflare", "@cf/meta/llama-2-7b-chat-int8", "FAIL/@cf/meta/llama-2-7b-chat-int8"),
    # ("cohere", "cohere", "command-light", "FAIL/command-light"),
    # ("google", "google", "gemini-1.5-flash", "FAIL/gemini-1.5-flash"),
    ("Cerebras", "Cerebras", "llama3.1-8b", "FAIL/llama3.1-8b"),
    ("SambaNova", "SambaNova", "Meta-Llama-3.1-8B-Instruct", "FAIL/Meta-Llama-3.1-8B-Instruct"),
    ("deepinfra", "deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct", "microsoft/WizardLM-2-8x22B-model-fail"),
    ("deepseek", "deepseek", "deepseek-chat", "FAIL/deepseek-chat"),
    # ("parasail", "parasail", "parasail-mistral-nemo", "parasail-mistral-nemo-fail"),
    ("x.ai", "x.ai", "grok-3-mini", "grok-3-mini-fail"),
    ("together.ai", "together.ai", "Qwen/Qwen3-Next-80B-A3B-Instruct", "meta-llama/Llama-3-8b-chat-hf-fail"),
    ("openrouter", "openrouter", "mistralai/mistral-nemo", "mistralai/mistral-nemo-fail"),
    # ("novita.ai", "novita.ai", "mistralai/mistral-nemo", "FAIL/mistralai/mistral-nemo"),
    ("mistral", "mistral", "open-mistral-7b", "FAIL/open-mistral-7b"),
    ("hyperbolic", "hyperbolic", "meta-llama/Meta-Llama-3.1-8B-Instruct", "FAIL/meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("groq", "groq", "qwen/qwen3-32b", "FAIL/llama3-8b-8192"),
    ("fireworks.ai", "fireworks.ai", "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
     "accounts/fireworks/models/llama4-scout-instruct-basic-fail"),
]
def _build_chat():
    c = ModelChat()
    c.add_user_message("What is the weather like in Paris today?")
    return c


def _keys_for(provider_keys, provider, key_name):
    return get_provider_key(provider_keys, provider, key_name)


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    TEST_PROVIDERS,
    ids=[p[0] for p in TEST_PROVIDERS],
)
def test_function_and_tool_mapping(provider_keys, provider, key_name, model, fail_model):
    """
    Test REAL LLM API calls with function calling.
    This is an integration test - NO mocks, real API requests.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = _keys_for(provider_keys, provider, key_name)
    chat = _build_chat()

    # REAL API call - creates actual HTTP request to provider
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)

    # Validate response structure (unified format across providers)
    assert res is not None, "Response should not be None"
    assert res.id is not None, "Response should have an ID"
    assert res.model is not None, "Response should have model name"

    assert_forced_weather_response(res, "get_weather")


def test_function_and_tool_fallback(provider_keys):
    """
    Test REAL fallback mechanism with function calling.
    Primary model fails → fallback to secondary model.
    Both are REAL API calls, no mocks.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}

    # Create fallback client (Anthropic) - REAL
    keys_anthropic = _keys_for(provider_keys, 'anthropic', 'anthropic')
    client_fallback = MagicLLM(model='claude-3-haiku-20240307', **keys_anthropic)

    # Create primary client with invalid model to trigger fallback - REAL
    keys_openai = _keys_for(provider_keys, 'openai', 'openai')
    client = MagicLLM(model='gpt-4o1', fallback=client_fallback, **keys_openai)

    chat = _build_chat()
    res: ModelChatResponse = client.llm.generate(chat, tools=[FUNCTION_DEF], tool_choice=tool_entry)

    # Validate response came from fallback (Anthropic)
    assert res is not None, "Response should not be None"
    assert res.model is not None, "Response should have model name"
    # Should have used Claude (fallback) since gpt-4o1 doesn't exist
    assert "claude" in res.model.lower(), f"Expected fallback to Claude, got model: {res.model}"

    assert_forced_weather_response(res, "get_weather")


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    TEST_PROVIDERS,
    ids=[p[0] for p in TEST_PROVIDERS],
)
def test_function_and_tool_mapping_stream(provider_keys, provider, key_name, model, fail_model):
    """
    Test REAL streaming LLM API calls with function calling.
    This is an integration test - NO mocks, real streaming HTTP requests.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = _keys_for(provider_keys, provider, key_name)
    chat = _build_chat()

    # REAL API call - creates actual streaming HTTP request
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)

    assert_forced_weather_stream(client.llm.stream_generate(chat), "get_weather")
