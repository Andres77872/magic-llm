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


# ─── Task: max_tokens → max_completion_tokens for official OpenAI ─────

@pytest.mark.parametrize("base_url,expect_rename", [
    ("https://api.openai.com/v1", True),
    ("https://api.openai.com/v1/", True),
    ("https://API.OPENAI.COM/v1", True),
    ("https://api.openai.com", True),
    ("https://api.openai.com/", True),
    ("https://api.openai.com/v1/models", False),
    ("https://api.groq.com/openai/v1", False),
    ("https://api.deepinfra.com/v1/openai", False),
    ("https://openrouter.ai/api/v1", False),
])
def test_openai_max_tokens_rename(chat_simple, base_url, expect_rename):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-5.4", base_url=base_url)
    body_bytes, _ = prov.transform_request(chat_simple, max_tokens=512)
    body = _decode_body(body_bytes)
    if expect_rename:
        assert "max_tokens" not in body, f"Expected max_tokens removed for {base_url}"
        assert body.get("max_completion_tokens") == 512, f"Expected max_completion_tokens=512 for {base_url}"
    else:
        assert body.get("max_tokens") == 512, f"Expected max_tokens preserved for {base_url}"
        assert "max_completion_tokens" not in body, f"Expected no max_completion_tokens for {base_url}"


def test_openai_max_completion_tokens_already_present_not_overwritten(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-5.4", base_url="https://api.openai.com/v1")
    body_bytes, _ = prov.transform_request(chat_simple, max_tokens=512, max_completion_tokens=1024)
    body = _decode_body(body_bytes)
    assert body.get("max_completion_tokens") == 1024, "Existing max_completion_tokens must not be overwritten"
    assert "max_tokens" not in body, "max_tokens must be removed when max_completion_tokens already set"


def test_openai_streaming_renames_max_tokens_and_adds_stream_options(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-5.4", base_url="https://api.openai.com/v1")
    body_bytes, _ = prov.transform_request(chat_simple, max_tokens=512, stream=True)
    body = _decode_body(body_bytes)
    assert "max_tokens" not in body
    assert body.get("max_completion_tokens") == 512
    assert body.get("stream_options") == {"include_usage": True}


def test_openai_non_streaming_no_max_tokens_no_op(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-5.4", base_url="https://api.openai.com/v1")
    body_bytes, _ = prov.transform_request(chat_simple, temperature=0.5)
    body = _decode_body(body_bytes)
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert body.get("temperature") == 0.5


def test_non_openai_provider_keeps_max_tokens(chat_simple):
    from magic_llm.engine.openai_adapters import ProviderGroq
    prov = ProviderGroq(api_key="gsk-xxx", model="mixtral-8x7b-32768")
    body_bytes, _ = prov.transform_request(chat_simple, max_tokens=512)
    body = _decode_body(body_bytes)
    assert body.get("max_tokens") == 512
    assert "max_completion_tokens" not in body


# ─── json_output/json_mode → response_format mapping ───────────────────


def test_openai_json_output_true_adds_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_output=True)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "json_output must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "json_output=True must map to response_format=json_object"
    )


def test_openai_json_mode_true_adds_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_mode=True)
    body = _decode_body(body_bytes)
    assert "json_mode" not in body, "json_mode must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "json_mode=True must map to response_format=json_object"
    )


def test_openai_json_output_false_stripped_no_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_output=False)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "json_output=False must be stripped"
    assert "response_format" not in body, (
        "json_output=False must NOT add response_format"
    )


def test_openai_json_mode_false_stripped_no_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_mode=False)
    body = _decode_body(body_bytes)
    assert "json_mode" not in body, "json_mode=False must be stripped"
    assert "response_format" not in body, (
        "json_mode=False must NOT add response_format"
    )


def test_openai_json_output_true_from_constructor_adds_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o", json_output=True)
    body_bytes, _ = prov.prepare_data(chat_simple)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "json_output from constructor must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "json_output=True from constructor must map to response_format"
    )


def test_openai_json_mode_true_from_constructor_adds_response_format(chat_simple):
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o", json_mode=True)
    body_bytes, _ = prov.prepare_data(chat_simple)
    body = _decode_body(body_bytes)
    assert "json_mode" not in body, "json_mode from constructor must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "json_mode=True from constructor must map to response_format"
    )


def test_openai_preserves_explicit_response_format(chat_simple):
    rf = {"type": "json_object"}
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, response_format=rf)
    body = _decode_body(body_bytes)
    assert body.get("response_format") == rf, "explicit response_format must be preserved"


def test_openai_json_output_true_does_not_overwrite_explicit_response_format(chat_simple):
    rf = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {"type": "object"}}}
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_output=True, response_format=rf)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "json_output stripped even with response_format present"
    assert body.get("response_format") == rf, "explicit response_format must NOT be overwritten"


def test_openai_json_mode_true_does_not_overwrite_explicit_response_format(chat_simple):
    rf = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {"type": "object"}}}
    prov = ProviderOpenAI(api_key="sk-xxx", model="gpt-4o")
    body_bytes, _ = prov.prepare_data(chat_simple, json_mode=True, response_format=rf)
    body = _decode_body(body_bytes)
    assert "json_mode" not in body, "json_mode stripped even with response_format present"
    assert body.get("response_format") == rf, "explicit response_format must NOT be overwritten"


def test_deepinfra_json_output_true_adds_response_format(chat_simple):
    prov = ProviderDeepInfra(api_key="sk-xxx", model="meta-llama/Meta-Llama-3.1-70B-Instruct")
    body_bytes, _ = prov.prepare_data(chat_simple, json_output=True, json_mode=True)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "DeepInfra: json_output must be stripped"
    assert "json_mode" not in body, "DeepInfra: json_mode must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "DeepInfra: json_output=True must map to response_format"
    )


def test_deepinfra_json_output_true_from_constructor_adds_response_format(chat_simple):
    prov = ProviderDeepInfra(
        api_key="sk-xxx",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        json_output=True,
        json_mode=True,
    )
    body_bytes, _ = prov.prepare_data(chat_simple)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "DeepInfra: json_output from constructor must be stripped"
    assert "json_mode" not in body, "DeepInfra: json_mode from constructor must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "DeepInfra: json_output=True from constructor must map to response_format"
    )


def test_sambanova_json_output_true_adds_response_format_via_base_class(chat_simple):
    prov = ProviderSambaNova(api_key="sk-xxx", model="Meta-Llama-3.1-8B-Instruct")
    body_bytes, _ = prov.prepare_data(chat_simple, json_output=True)
    body = _decode_body(body_bytes)
    assert "json_output" not in body, "SambaNova (via base): json_output must be stripped"
    assert body.get("response_format") == {"type": "json_object"}, (
        "SambaNova (via base): json_output=True must map to response_format"
    )
