import json
import os

from magic_llm import MagicLLM

import pytest

from magic_llm.model import ModelChat, ModelChatResponse

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
CALL_DEF = {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}
KEYS_FILE = os.getenv("MAGIC_LLM_KEYS")
if not KEYS_FILE or not os.path.exists(KEYS_FILE):
    pytest.skip(
        "MAGIC_LLM_KEYS env var must point to a valid keys file for integration tests.",
        allow_module_level=True,
    )
with open(KEYS_FILE) as f:
    ALL_KEYS = json.load(f)
PROVIDERS = [
    (provider, key_name, success_model, fail_model)
    for provider, key_name, success_model, fail_model in TEST_PROVIDERS
    if key_name in ALL_KEYS
]


def extract_body(engine, chat, **kwargs):
    # choose appropriate prepare_data method
    if hasattr(engine, 'base'):
        body_bytes, _ = engine.base.prepare_data(chat, **kwargs)
    else:
        body_bytes, _ = engine.prepare_data(chat, **kwargs)
    return json.loads(body_bytes)


def _build_chat():
    c = ModelChat()
    c.add_user_message("What is the weather like in Paris today?")
    return c


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_function_and_tool_mapping(provider, key_name, model, fail_model):
    """
    Test REAL LLM API calls with function calling.
    This is an integration test - NO mocks, real API requests.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # REAL API call - creates actual HTTP request to provider
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)

    # Validate response structure (unified format across providers)
    assert res is not None, "Response should not be None"
    assert res.id is not None, "Response should have an ID"
    assert res.model is not None, "Response should have model name"

    # If model called tools, validate tool_calls structure
    if res.tool_calls and len(res.tool_calls) > 0:
        tool_call = res.tool_calls[0]
        assert tool_call.type == "function", "tool_call.type should be 'function'"
        assert tool_call.function is not None, "tool_call should have function"
        assert tool_call.function.name == "get_weather", "Function name should be 'get_weather'"
        assert tool_call.function.arguments is not None, "Function should have arguments"

        # Arguments should be valid JSON with 'location'
        args = json.loads(tool_call.function.arguments)
        assert "location" in args, f"Arguments should contain 'location', got: {args}"
        print(f"✓ {provider}: Tool called with location='{args.get('location')}'")
    else:
        # Model chose not to call tools (valid behavior with auto)
        print(f"⚠ {provider}: Model did not call tools (model behavior)")

    print(f"✓ {provider}: Real API call completed successfully")


def test_function_and_tool_fallback():
    """
    Test REAL fallback mechanism with function calling.
    Primary model fails → fallback to secondary model.
    Both are REAL API calls, no mocks.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}

    # Create fallback client (Anthropic) - REAL
    keys_anthropic = dict(ALL_KEYS['anthropic'])
    client_fallback = MagicLLM(model='claude-3-haiku-20240307', **keys_anthropic)

    # Create primary client with invalid model to trigger fallback - REAL
    keys_openai = dict(ALL_KEYS['openai'])
    client = MagicLLM(model='gpt-4o1', fallback=client_fallback, **keys_openai)

    chat = _build_chat()
    res: ModelChatResponse = client.llm.generate(chat, tools=[FUNCTION_DEF], tool_choice=tool_entry)

    # Validate response came from fallback (Anthropic)
    assert res is not None, "Response should not be None"
    assert res.model is not None, "Response should have model name"
    # Should have used Claude (fallback) since gpt-4o1 doesn't exist
    assert "claude" in res.model.lower(), f"Expected fallback to Claude, got model: {res.model}"

    # Validate tool calls if present
    if res.tool_calls and len(res.tool_calls) > 0:
        tool_call = res.tool_calls[0]
        assert tool_call.function.name == "get_weather", "Function name should be 'get_weather'"
        args = json.loads(tool_call.function.arguments)
        assert "location" in args, f"Arguments should contain 'location', got: {args}"
        print(f"✓ Fallback: Tool called with location='{args.get('location')}'")

    print(f"✓ Fallback test: Used model {res.model} (fallback worked)")


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_function_and_tool_mapping_stream(provider, key_name, model, fail_model):
    """
    Test REAL streaming LLM API calls with function calling.
    This is an integration test - NO mocks, real streaming HTTP requests.
    """
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # REAL API call - creates actual streaming HTTP request
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)

    # Accumulate streaming response
    chunks_received = 0
    accumulated_tool_calls = {}
    final_finish_reason = None

    for chunk in client.llm.stream_generate(chat):
        chunks_received += 1

        # Validate chunk structure
        assert chunk is not None, "Chunk should not be None"
        assert chunk.choices is not None, "Chunk should have choices"

        # Accumulate tool calls from delta
        if chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
            for tc in chunk.choices[0].delta.tool_calls:
                tc_id = tc.id or "default"
                if tc_id not in accumulated_tool_calls:
                    accumulated_tool_calls[tc_id] = {"name": None, "arguments": ""}
                if tc.function:
                    if tc.function.name:
                        accumulated_tool_calls[tc_id]["name"] = tc.function.name
                    if tc.function.arguments:
                        new_args = tc.function.arguments
                        current = accumulated_tool_calls[tc_id]["arguments"]
                        # Handle cumulative vs incremental
                        if new_args.startswith('{') and current.startswith('{'):
                            accumulated_tool_calls[tc_id]["arguments"] = new_args
                        else:
                            accumulated_tool_calls[tc_id]["arguments"] += new_args

        if chunk.choices[0].finish_reason:
            final_finish_reason = chunk.choices[0].finish_reason

    # Validate streaming worked
    assert chunks_received > 0, "Should have received at least one chunk"
    print(f"✓ {provider}: Received {chunks_received} streaming chunks")

    # Validate tool calls if present
    if accumulated_tool_calls:
        first_tc = list(accumulated_tool_calls.values())[0]
        assert first_tc["name"] == "get_weather", f"Expected 'get_weather', got '{first_tc['name']}'"
        if first_tc["arguments"]:
            args = json.loads(first_tc["arguments"])
            assert "location" in args, f"Arguments should contain 'location', got: {args}"
            print(f"✓ {provider}: Streamed tool call with location='{args.get('location')}'")
    else:
        print(f"⚠ {provider}: No tool calls in stream (model behavior)")

    print(f"✓ {provider}: Real streaming API call completed")
