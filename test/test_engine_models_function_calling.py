import json
import os
import sys

from magic_llm import MagicLLM

# add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from magic_llm.model import ModelChat, ModelChatResponse

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
    ("parasail", "parasail", "parasail-mistral-nemo", "parasail-mistral-nemo-fail"),
    ("x.ai", "x.ai", "grok-3-mini", "grok-3-mini-fail"),
    ("together.ai", "together.ai", "meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-8b-chat-hf-fail"),
    ("perplexity", "perplexity", "sonar", "sonar-fail"),
    ("openrouter", "openrouter", "mistralai/mistral-nemo", "mistralai/mistral-nemo-fail"),
    # ("novita.ai", "novita.ai", "mistralai/mistral-nemo", "FAIL/mistralai/mistral-nemo"),
    ("mistral", "mistral", "open-mistral-7b", "FAIL/open-mistral-7b"),
    ("hyperbolic", "hyperbolic", "meta-llama/Meta-Llama-3.1-8B-Instruct", "FAIL/meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("groq", "groq", "llama3-8b-8192", "FAIL/llama3-8b-8192"),
    ("fireworks.ai", "fireworks.ai", "accounts/fireworks/models/llama4-scout-instruct-basic",
     "accounts/fireworks/models/llama4-scout-instruct-basic-fail"),
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


def test_function_and_tool_fallback():
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = dict(ALL_KEYS['anthropic'])
    chat = _build_chat()
    client_fallback = MagicLLM(model='claude-3-haiku-20240307', **keys)
    keys = dict(ALL_KEYS['openai'])
    client = MagicLLM(model='gpt-4o1', fallback=client_fallback, **keys)
    res: ModelChatResponse = client.llm.generate(chat, tools=[FUNCTION_DEF], tool_choice=tool_entry)

    print(res.model)
    print(res.tool_calls)


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_function_and_tool_mapping_stream(provider, key_name, model, fail_model):
    # Unified OpenAI style: tools + tool_choice
    tool_entry = {"type": "function", "function": {"name": FUNCTION_DEF['function']['name']}}
    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()
    client = MagicLLM(model=model, tools=[FUNCTION_DEF], tool_choice=tool_entry, **keys)
    res = client.llm.stream_generate(chat)
    for chunk in res:
        print(chunk)
