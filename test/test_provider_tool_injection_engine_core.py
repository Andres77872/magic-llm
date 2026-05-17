"""Regression tests for engine/core-owned provider tool injection."""

import json
import pytest

from magic_llm.agent.types import ToolResult
from magic_llm.engine.engine_amazon import EngineAmazon
from magic_llm.engine.engine_anthropic import EngineAnthropic
from magic_llm.engine.engine_azure import EngineAzure
from magic_llm.engine.engine_cloudflare import EngineCloudFlare
from magic_llm.engine.engine_cohere import EngineCohere
from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.engine.openai_adapters import ProviderDeepInfra, ProviderOpenAI
from magic_llm.engine.tooling import (
    AnthropicStreamState,
    append_tool_results,
    guard_tools_supported,
    map_request_tools,
    validate_tool_result_integrity,
)
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import UsageModel


def _chat() -> ModelChat:
    chat = ModelChat()
    chat.add_user_message("Use a tool")
    return chat


def _decode(body: bytes) -> dict:
    return json.loads(body.decode("utf-8"))


def _get_weather(city: str) -> str:
    """Get weather for a city."""
    return "sunny"


def _legacy_tool() -> dict:
    return {
        "name": "get_weather",
        "description": "Get weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


class TestExactlyOnceProviderMapping:
    def test_openai_provider_maps_raw_tools_once_via_engine_core(self, monkeypatch):
        provider = ProviderOpenAI(api_key="sk-test", model="gpt-4o")
        calls = []

        def spy(provider_key, tools, tool_choice):
            calls.append((provider_key, tools, tool_choice))
            return map_request_tools(provider_key, tools, tool_choice)

        monkeypatch.setattr("magic_llm.engine.openai_adapters.base_provider.map_request_tools", spy)

        body = _decode(provider.prepare_data(_chat(), tools=[_get_weather], tool_choice={"name": "get_weather"})[0])

        assert len(calls) == 1
        assert calls[0] == ("openai", [_get_weather], {"name": "get_weather"})
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "_get_weather"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_anthropic_provider_maps_raw_tools_once_via_engine_core(self, monkeypatch):
        engine = EngineAnthropic(api_key="ak-test", model="claude-3-haiku-20240307")
        calls = []

        def spy(provider_key, tools, tool_choice):
            calls.append((provider_key, tools, tool_choice))
            return map_request_tools(provider_key, tools, tool_choice)

        monkeypatch.setattr("magic_llm.engine.engine_anthropic.map_request_tools", spy)

        body = _decode(engine.prepare_data(_chat(), tools=[_get_weather], tool_choice={"name": "get_weather"})[0])

        assert len(calls) == 1
        assert calls[0] == ("anthropic", [_get_weather], {"name": "get_weather"})
        assert body["tools"][0]["name"] == "_get_weather"
        assert body["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_gemini_provider_maps_raw_tools_once_via_engine_core(self, monkeypatch):
        engine = EngineGoogle(api_key="g-test", model="gemini-2.5-flash")
        calls = []

        def spy(provider_key, tools, tool_choice):
            calls.append((provider_key, tools, tool_choice))
            return map_request_tools(provider_key, tools, tool_choice)

        monkeypatch.setattr("magic_llm.engine.engine_google.map_request_tools", spy)

        body = engine.prepare_data_sync(_chat(), tools=[_get_weather], tool_choice={"name": "get_weather"})[2]

        assert len(calls) == 1
        assert calls[0] == ("google", [_get_weather], {"name": "get_weather"})
        assert body["tools"][0]["functionDeclarations"][0]["name"] == "_get_weather"
        assert body["toolConfig"] == {
            "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
        }

    def test_deepinfra_preserves_named_tool_choice(self):
        provider = ProviderDeepInfra(api_key="sk-test", model="meta-llama/Meta-Llama-3.1-70B-Instruct")

        body = _decode(provider.prepare_data(_chat(), tools=[_legacy_tool()], tool_choice={"name": "get_weather"})[0])

        assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
        assert body["tool_choice"] != "auto"

    def test_provider_shaped_dict_tools_are_not_required_for_gemini(self):
        engine = EngineGoogle(api_key="g-test", model="gemini-2.5-flash")

        body = engine.prepare_data_sync(_chat(), tools=[_get_weather], tool_choice="auto")[2]

        assert body["tools"][0]["functionDeclarations"][0]["name"] == "_get_weather"


class TestToolResultMessages:
    def test_openai_tool_results_are_role_tool_and_include_errors(self):
        chat = _chat()
        results = [ToolResult(tool_call_id="call_1", name="get_weather", content="timeout", is_error=True)]

        append_tool_results("openai", chat, results)

        assert chat.messages[-1]["role"] == "tool"
        assert chat.messages[-1]["tool_call_id"] == "call_1"
        assert chat.messages[-1]["content"] == "timeout"
        assert chat.messages[-1]["is_error"] is True

    def test_anthropic_tool_results_bundle_and_validate_completeness(self):
        chat = _chat()
        chat.add_tool_call_message(tool_calls=[
            {"id": "call_1", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "tool_b", "arguments": "{}"}},
        ])

        with pytest.raises(ValueError, match="missing.*call_2"):
            append_tool_results("anthropic", chat, [ToolResult(tool_call_id="call_1", name="tool_a", content="a")])

        append_tool_results("anthropic", chat, [
            ToolResult(tool_call_id="call_1", name="tool_a", content="a"),
            ToolResult(tool_call_id="call_2", name="tool_b", content="b", is_error=True),
        ])

        msg = chat.messages[-1]
        assert msg["role"] == "user"
        assert [part["tool_use_id"] for part in msg["content"]] == ["call_1", "call_2"]
        assert validate_tool_result_integrity("anthropic", chat) is True

    def test_gemini_tool_results_use_function_response_parts_and_error_shape(self):
        chat = _chat()
        chat.add_tool_call_message(tool_calls=[
            {"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}},
        ])

        append_tool_results("google", chat, [
            ToolResult(tool_call_id="call_1", name="get_weather", content="broken", is_error=True)
        ])

        fr = chat.messages[-1]["content"][0]["functionResponse"]
        assert fr == {"id": "call_1", "name": "get_weather", "response": {"error": "broken"}}
        assert validate_tool_result_integrity("google", chat) is True


class TestAnthropicStreamStateIsolation:
    def test_interleaved_stream_states_do_not_cross_contaminate(self):
        engine = EngineAnthropic(api_key="ak-test", model="claude-3-haiku-20240307")
        state_a = AnthropicStreamState()
        state_b = AnthropicStreamState()

        start_a = {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tool_a", "name": "alpha", "input": {}}}
        start_b = {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tool_b", "name": "beta", "input": {}}}
        delta_a = {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city":"A"}'}}
        delta_b = {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city":"B"}'}}
        stop_a = {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}}
        stop_b = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}

        usage_a = UsageModel(prompt_tokens=1, completion_tokens=0, total_tokens=1)
        usage_b = UsageModel(prompt_tokens=1, completion_tokens=0, total_tokens=1)
        engine.prepare_chunk(start_a, "msg_a", usage_a, state_a)
        engine.prepare_chunk(start_b, "msg_b", usage_b, state_b)
        chunk_a, _, usage_a = engine.prepare_chunk(delta_a, "msg_a", usage_a, state_a)
        chunk_b, _, usage_b = engine.prepare_chunk(delta_b, "msg_b", usage_b, state_b)
        engine.prepare_chunk(stop_a, "msg_a", usage_a, state_a)
        engine.prepare_chunk(stop_b, "msg_b", usage_b, state_b)

        call_a = chunk_a.choices[0].delta.tool_calls[0]
        call_b = chunk_b.choices[0].delta.tool_calls[0]
        assert call_a.id == "tool_a"
        assert call_a.function.name == "alpha"
        assert call_a.function.arguments == '{"city":"A"}'
        assert call_b.id == "tool_b"
        assert call_b.function.name == "beta"
        assert call_b.function.arguments == '{"city":"B"}'
        assert state_a.finish_reason == "tool_calls"
        assert state_b.finish_reason == "stop"


class TestIncompleteToolingGuardrails:
    @pytest.mark.parametrize("engine_name", ["Cohere", "Cloudflare", "Amazon Bedrock", "Amazon Bedrock Anthropic", "Amazon Bedrock Nova", "Azure"])
    def test_guardrail_message_is_package_wiring_specific(self, engine_name):
        with pytest.raises(ChatException) as exc_info:
            guard_tools_supported(engine_name, tools=[_get_weather], tool_choice=None)

        assert exc_info.value.error_code == "TOOL_WIRING_NOT_IMPLEMENTED"
        assert engine_name in str(exc_info.value)
        assert "package-side wiring limitation" in str(exc_info.value)

    def test_guardrail_allows_plain_chat(self):
        guard_tools_supported("Cohere", tools=None, tool_choice=None)
        guard_tools_supported("Cloudflare", tools=[], tool_choice="none")

    def test_cohere_plain_prepare_data_unchanged_but_tools_fail(self):
        engine = EngineCohere(api_key="co-test", model="command-r")
        assert _decode(engine.prepare_data(_chat())[0])["message"] == "Use a tool"
        with pytest.raises(ChatException, match="Cohere"):
            engine.prepare_data(_chat(), tools=[_get_weather])

    def test_cloudflare_plain_prepare_data_unchanged_but_tool_choice_fails(self):
        engine = EngineCloudFlare(api_key="cf-test", account_id="acct", model="model")
        assert _decode(engine.prepare_data(_chat())[0])["messages"][-1]["content"] == "Use a tool"
        with pytest.raises(ChatException, match="Cloudflare"):
            engine.prepare_data(_chat(), tool_choice="auto")

    def test_amazon_and_azure_tool_guardrails_fire_before_unimplemented_paths(self):
        amazon = EngineAmazon(api_key="unused", aws_access_key_id="x", aws_secret_access_key="y", model="amazon.nova-lite")
        azure = EngineAzure(speech_key="key", speech_region="eastus", model="tts")

        with pytest.raises(ChatException, match="Amazon Bedrock"):
            amazon.transform_request(_chat(), tools=[_get_weather])
        with pytest.raises(ChatException, match="Azure"):
            azure.generate(_chat(), tools=[_get_weather])
