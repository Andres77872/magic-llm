import json
import os
import sys

from magic_llm import MagicLLM
from magic_llm.util.tools_mapping import normalize_openai_tools

# add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from magic_llm.model import ModelChat

try:
    from pydantic import BaseModel  # v2
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore

# Limit to a representative set of real providers we support end-to-end
TEST_PROVIDERS = [
    ("openai", "openai", "gpt-4o", "gpt-4o1"),
    ("anthropic", "anthropic", "claude-3-haiku-20240307", "FAIL/claude-3-haiku-20240307"),
    ("deepinfra", "deepinfra", "meta-llama/Meta-Llama-3.1-70B-Instruct", "microsoft/WizardLM-2-8x22B-model-fail"),
]

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


def _build_chat():
    c = ModelChat()
    c.add_user_message("Please check the weather for Bogotá and maybe use the tools if needed.")
    return c


def _python_tool_definitions():
    def get_weather(location: str):
        """
        Retrieve current weather information for a given location.

        This function fetches and returns weather details based on the
        specified location. The location is required and must be a valid
        string indicating the place for which the weather is queried.
        Ensure the location provided is recognized by the weather API used
        by the function.

        :param location: The name of the location to retrieve weather
            information for.
        :type location: str
        :return: Weather details for the specified location.
        :rtype: str
        """
        return ""

    tools = [get_weather]

    if BaseModel is not None:
        class GetForecast(BaseModel):
            """Forecast for a given location and days."""
            location: str
            days: int

        tools.append(GetForecast)

    return tools, "get_weather"


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_at_init(provider, key_name, model, _fail_model):
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # Pass tools at initialization time
    client = MagicLLM(model=model, tools=tools, tool_choice=tool_entry, **keys)
    res = client.llm.generate(chat)
    print(res)


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_call_time_override(provider, key_name, model, _fail_model):
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # Provide some defaults at init, override at call time
    client = MagicLLM(model=model, tools=tools[:1], tool_choice="auto", **keys)

    # Now override both tools and tool_choice at call time
    res = client.llm.generate(chat, tools=tools, tool_choice=tool_entry)
    print(res)


@pytest.mark.parametrize(
    ("provider", "key_name", "model", "_fail_model"),
    PROVIDERS,
    ids=[p[0] for p in PROVIDERS],
)
def test_python_and_pydantic_tools_at_init_stream(provider, key_name, model, _fail_model):
    tools, primary_name = _python_tool_definitions()
    tool_entry = {"type": "function", "function": {"name": primary_name}}

    keys = dict(ALL_KEYS[key_name])
    chat = _build_chat()

    # Pass tools at initialization time
    client = MagicLLM(model=model, tools=tools, tool_choice=tool_entry, **keys)
    for i in client.llm.stream_generate(chat):
        print(i)

def test_to_json():
    # Pass tools at initialization time
    tools, primary_name = _python_tool_definitions()
    res =normalize_openai_tools([tools[0]])
    print(res)