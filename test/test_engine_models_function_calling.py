import json
import os
import sys

from magic_llm import MagicLLM

# add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from magic_llm.model import ModelChat

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
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307")
]
CALL_DEF = {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}
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
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)
    print(res)
