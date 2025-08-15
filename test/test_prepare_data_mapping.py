import json
import pytest

from magic_llm.model import ModelChat
from magic_llm.engine.openai_adapters import ProviderOpenAI, ProviderDeepInfra
from magic_llm.engine.engine_anthropic import EngineAnthropic


LEGACY_FUNC = {
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"],
    },
}


@pytest.fixture()
def chat_simple():
    c = ModelChat()
    c.add_user_message("What is the weather like?")
    return c


def _decode_body(body_bytes):
    assert isinstance(body_bytes, (bytes, bytearray))
    return json.loads(body_bytes.decode("utf-8"))


def test_openai_provider_prepare_data_maps_tools_and_choice(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")

    body_bytes, _ = prov.prepare_data(chat_simple, tools=[LEGACY_FUNC], tool_choice={"name": "get_weather"})
    body = _decode_body(body_bytes)

    # tools normalized to OpenAI function wrapper
    assert "tools" in body and isinstance(body["tools"], list) and len(body["tools"]) == 1
    t0 = body["tools"][0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "get_weather"
    assert "parameters" in t0["function"]

    # tool_choice normalized to OpenAI format
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_deepinfra_provider_coerces_tool_choice_to_string(chat_simple):
    prov = ProviderDeepInfra(api_key="sk-xxx", model="meta-llama/Meta-Llama-3.1-70B-Instruct")

    body_bytes, _ = prov.prepare_data(chat_simple, tools=[LEGACY_FUNC], tool_choice={"name": "get_weather"})
    body = _decode_body(body_bytes)

    # tools normalized to OpenAI function wrapper
    assert isinstance(body.get("tools"), list) and body["tools"][0]["type"] == "function"

    # tool_choice coerced to a string (default 'auto')
    assert body.get("tool_choice") == "auto"


@pytest.mark.parametrize(
    "choice_in, expected",
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", None),
        ({"name": "get_weather"}, {"type": "tool", "name": "get_weather"}),
    ],
)
def test_anthropic_engine_prepare_data_maps_tools_and_choice(chat_simple, choice_in, expected):
    eng = EngineAnthropic(api_key="ak-xxx", model="claude-3-haiku-20240307")

    body_bytes, headers = eng.prepare_data(chat_simple, tools=[LEGACY_FUNC], tool_choice=choice_in)
    body = _decode_body(body_bytes)

    # tools mapped to Anthropic schema with input_schema
    assert isinstance(body.get("tools"), list)
    t0 = body["tools"][0]
    assert t0["name"] == "get_weather" and "input_schema" in t0

    # tool_choice mapped as expected
    assert body.get("tool_choice") == expected

    # sanity: messages exist
    assert isinstance(body.get("messages"), list) and len(body["messages"]) >= 1
