import json
import pytest

from magic_llm.model import ModelChat
from magic_llm.engine.openai_adapters import (
    ProviderOpenAI,
    ProviderDeepInfra,
    ProviderSambaNova,
)
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


def test_openai_provider_prepare_data_accepts_callable(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")

    def get_weather(location: str):
        """Get current temperature for a given location."""
        return ""

    body_bytes, _ = prov.prepare_data(chat_simple, tools=[get_weather], tool_choice={"name": "get_weather"})
    body = _decode_body(body_bytes)

    # tools normalized to OpenAI function wrapper
    assert "tools" in body and isinstance(body["tools"], list) and len(body["tools"]) == 1
    t0 = body["tools"][0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "get_weather"
    assert "parameters" in t0["function"]
    # tool_choice normalized to OpenAI format
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_deepinfra_provider_preserves_named_tool_choice(chat_simple):
    prov = ProviderDeepInfra(api_key="sk-xxx", model="meta-llama/Meta-Llama-3.1-70B-Instruct")

    body_bytes, _ = prov.prepare_data(chat_simple, tools=[LEGACY_FUNC], tool_choice={"name": "get_weather"})
    body = _decode_body(body_bytes)

    # tools normalized to OpenAI function wrapper
    assert isinstance(body.get("tools"), list) and body["tools"][0]["type"] == "function"

    # Named tool_choice intent is preserved; it is never silently downgraded to auto.
    assert body.get("tool_choice") == {"type": "function", "function": {"name": "get_weather"}}


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


# ─── Task 7.6: is_error hygiene ──────────────────────────────────────────────

def _chat_with_tool_error() -> ModelChat:
    """Build a ModelChat with a tool result that has is_error=True."""
    chat = ModelChat()
    chat.add_user_message("What's the weather?")
    chat.add_assistant_message("Let me check.")
    chat.add_tool_result(
        tool_call_id="call_abc",
        content='{"error": "API unreachable", "type": "ConnectionError"}',
        is_error=True,
    )
    chat.add_tool_result(
        tool_call_id="call_def",
        content="sunny, 72°F",
        is_error=False,
    )
    return chat


def test_openai_provider_strips_is_error_from_tool_messages():
    """ProviderOpenAI: is_error field MUST be stripped from role=tool messages
    in the wire-format payload.

    The is_error field is stored in ModelChat.messages for internal debugging
    but is NOT part of the OpenAI-compatible spec. It MUST NOT appear in the
    serialized JSON sent to the provider.
    """
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(_chat_with_tool_error())
    body = _decode_body(body_bytes)

    tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 2, "Both tool results should be present"

    for msg in tool_messages:
        assert "is_error" not in msg, (
            f"is_error field must be stripped from tool messages, got: {msg}"
        )

    # Internal state still has is_error (we only strip at wire format)
    internal_messages = _chat_with_tool_error().get_messages()
    internal_tool = [m for m in internal_messages if m.get("role") == "tool"]
    assert internal_tool[0].get("is_error") is True, (
        "Internal ModelChat.messages still carries is_error for debugging"
    )
    assert internal_tool[1].get("is_error") is False, (
        "Internal ModelChat.messages still carries is_error for debugging"
    )


def test_deepinfra_provider_strips_is_error_from_tool_messages():
    """ProviderDeepInfra: is_error MUST be stripped from tool messages.

    NOTE: ProviderDeepInfra.prepare_data() has its own implementation that
    bypasses OpenAiBaseProvider.transform_request(). The fix is applied
    separately in ProviderDeepInfra.prepare_data().
    """
    prov = ProviderDeepInfra(
        api_key="sk-xxx",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
    body_bytes, _ = prov.prepare_data(_chat_with_tool_error())
    body = _decode_body(body_bytes)

    tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 2

    for msg in tool_messages:
        assert "is_error" not in msg, (
            f"is_error field must be stripped from tool messages for DeepInfra, got: {msg}"
        )


def test_sambanova_provider_strips_is_error_from_tool_messages():
    """ProviderSambaNova: is_error MUST be stripped from tool messages.

    ProviderSambaNova.prepare_data() calls super().prepare_data() which
    delegates to OpenAiBaseProvider.transform_request() — the base fix applies.
    """
    prov = ProviderSambaNova(
        api_key="sk-xxx",
        model="Meta-Llama-3.1-8B-Instruct",
    )
    body_bytes, _ = prov.prepare_data(_chat_with_tool_error())
    body = _decode_body(body_bytes)

    tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 2

    for msg in tool_messages:
        assert "is_error" not in msg, (
            f"is_error field must be stripped from tool messages for SambaNova, got: {msg}"
        )


def test_anthropic_engine_prepare_data_accepts_callable_and_pydantic(chat_simple):
    try:
        from pydantic import BaseModel
    except Exception:
        BaseModel = None

    def get_weather(location: str):
        """Get current temperature for a given location."""
        return ""

    tools = [get_weather]
    if BaseModel is not None:
        class GetForecast(BaseModel):
            """Forecast for a given location and days."""
            location: str
            days: int

        tools.append(GetForecast)

    eng = EngineAnthropic(api_key="ak-xxx", model="claude-3-haiku-20240307")
    body_bytes, headers = eng.prepare_data(chat_simple, tools=tools, tool_choice={"name": "get_weather"})
    body = _decode_body(body_bytes)

    assert isinstance(body.get("tools"), list) and len(body["tools"]) >= 1
    # First tool is callable get_weather
    assert any(t.get("name") == "get_weather" and "input_schema" in t for t in body["tools"]) 
