"""Unit tests for ToolAdapter protocol, adapters, factory, and re-exports.

Tests cover:
- Slice 5: ToolAdapter protocol definition
- Slices 6-7: OpenAIToolAdapter
- Slices 8-9: AnthropicToolAdapter
- Slice 10: ToolAdapterFactory
- Slice 11: Adapter package re-exports
"""

import json
from unittest.mock import MagicMock

import pytest

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, ToolCall, FunctionCall
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter, ToolAdapterFactory
from magic_llm.agent.adapters import OpenAIToolAdapter, AnthropicToolAdapter, GeminiToolAdapter


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
):
    """Build a valid ModelChatResponse."""
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="test-1",
        object="chat.completion",
        created=1700000000.0,
        model="test-model",
        choices=[choice],
    )


def _make_tool_call(id="call_1", name="get_weather", arguments='{"city":"London"}'):
    """Build a valid ToolCall."""
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))


# ─── Slice 5: ToolAdapter protocol definition ───────────────────────────────


class TestToolAdapterProtocol:
    """Slice 5: Protocol definition and runtime checkability."""

    def test_tool_adapter_protocol_is_runtime_checkable(self):
        """isinstance(some_adapter, ToolAdapter) works."""
        adapter = OpenAIToolAdapter()
        assert isinstance(adapter, ToolAdapter)

        adapter2 = AnthropicToolAdapter()
        assert isinstance(adapter2, ToolAdapter)

        adapter3 = GeminiToolAdapter()
        assert isinstance(adapter3, ToolAdapter)

    def test_tool_adapter_protocol_has_required_methods(self):
        """All 6 methods present."""
        required_methods = [
            "serialize_tool_defs",
            "deserialize_tool_calls",
            "serialize_tool_results",
            "is_finished",
            "extract_final_text",
            "validate_pair_integrity",
        ]
        for method_name in required_methods:
            assert hasattr(OpenAIToolAdapter, method_name), f"Missing {method_name}"
            assert hasattr(AnthropicToolAdapter, method_name), f"Missing {method_name}"
            assert hasattr(GeminiToolAdapter, method_name), f"Missing {method_name}"


# ─── Slice 6: OpenAIToolAdapter — serialize_tool_defs + deserialize_tool_calls


class TestOpenAISerializeToolDefs:
    """Slice 6: OpenAIToolAdapter.serialize_tool_defs."""

    def test_openai_serialize_tool_defs_callable(self):
        """Pass callable, assert {'type': 'function', 'function': {'name': ..., 'description': ..., 'parameters': ...}}."""

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return "sunny"

        adapter = OpenAIToolAdapter()
        result = adapter.serialize_tool_defs([get_weather])

        assert result is not None
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather for a city."
        assert "parameters" in tool["function"]

    def test_openai_serialize_tool_defs_dict(self):
        """Pass dict spec, assert passthrough or normalized."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
        adapter = OpenAIToolAdapter()
        result = adapter.serialize_tool_defs([tool_spec])

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"

    def test_openai_serialize_tool_defs_empty(self):
        """Empty tools list returns None."""
        adapter = OpenAIToolAdapter()
        result = adapter.serialize_tool_defs([])
        assert result is None


class TestOpenAIDeserializeToolCalls:
    """Slice 6: OpenAIToolAdapter.deserialize_tool_calls."""

    def test_openai_deserialize_tool_calls_extracts_from_response(self):
        """Construct ModelChatResponse with tool_calls, assert list[CanonicalToolCall] with parsed arguments dict."""
        tc = _make_tool_call(id="call_abc", name="get_weather", arguments='{"city":"London"}')
        response = _make_response(tool_calls=[tc])

        adapter = OpenAIToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 1
        call = result[0]
        assert isinstance(call, CanonicalToolCall)
        assert call.id == "call_abc"
        assert call.name == "get_weather"
        assert isinstance(call.arguments, dict)
        assert call.arguments["city"] == "London"

    def test_openai_deserialize_tool_calls_empty_response(self):
        """No tool_calls, assert empty list."""
        response = _make_response(content="Hello!")
        adapter = OpenAIToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert result == []

    def test_openai_deserialize_tool_calls_malformed_json(self):
        """Malformed JSON in arguments, assert empty dict."""
        tc = _make_tool_call(id="call_1", name="bad_tool", arguments="not json")
        response = _make_response(tool_calls=[tc])

        adapter = OpenAIToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 1
        assert result[0].arguments == {}


# ─── Slice 7: OpenAIToolAdapter — serialize_tool_results + is_finished + validate


class TestOpenAISerializeToolResults:
    """Slice 7: OpenAIToolAdapter.serialize_tool_results."""

    def test_openai_serialize_tool_results_appends_separate_tool_messages(self):
        """3 ToolResult objects, assert 3 separate role='tool' messages appended to chat.messages."""
        chat = ModelChat()
        chat.add_user_message("Hello")

        results = [
            ToolResult(tool_call_id="call_1", name="tool_a", content="result_a"),
            ToolResult(tool_call_id="call_2", name="tool_b", content="result_b"),
            ToolResult(tool_call_id="call_3", name="tool_c", content="result_c"),
        ]

        adapter = OpenAIToolAdapter()
        adapter.serialize_tool_results(results, chat)

        # Should have: user + 3 tool messages = 4 (no system message when ModelChat() has no system)
        assert len(chat.messages) == 4
        tool_msgs = [m for m in chat.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 3
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "result_a"
        assert tool_msgs[1]["tool_call_id"] == "call_2"
        assert tool_msgs[2]["tool_call_id"] == "call_3"


class TestOpenAIIsFinished:
    """Slice 7: OpenAIToolAdapter.is_finished."""

    def test_openai_is_finished_stop_returns_true(self):
        """finish_reason='stop', assert True."""
        response = _make_response(finish_reason="stop")
        adapter = OpenAIToolAdapter()
        assert adapter.is_finished(response) is True

    def test_openai_is_finished_tool_calls_returns_false(self):
        """finish_reason='tool_calls', assert False."""
        response = _make_response(finish_reason="tool_calls")
        adapter = OpenAIToolAdapter()
        assert adapter.is_finished(response) is False


class TestOpenAIValidatePairIntegrity:
    """Slice 7: OpenAIToolAdapter.validate_pair_integrity."""

    def test_openai_validate_pair_integrity_matched_returns_true(self):
        """Assistant with tool_calls + matching tool messages, assert True."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[{"id": "call_1", "function": {"name": "get_weather"}}]
        )
        chat.add_tool_result(tool_call_id="call_1", content="sunny")

        adapter = OpenAIToolAdapter()
        assert adapter.validate_pair_integrity(chat) is True

    def test_openai_validate_pair_integrity_orphaned_returns_false(self):
        """Assistant with 2 tool_calls but only 1 tool result, assert False."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool_a"}},
                {"id": "call_2", "function": {"name": "tool_b"}},
            ]
        )
        chat.add_tool_result(tool_call_id="call_1", content="result_a")

        adapter = OpenAIToolAdapter()
        assert adapter.validate_pair_integrity(chat) is False


class TestOpenAIExtractFinalText:
    """OpenAIToolAdapter.extract_final_text."""

    def test_openai_extract_final_text_returns_content(self):
        response = _make_response(content="Hello, world!")
        adapter = OpenAIToolAdapter()
        assert adapter.extract_final_text(response) == "Hello, world!"

    def test_openai_extract_final_text_empty(self):
        response = _make_response(content=None)
        adapter = OpenAIToolAdapter()
        assert adapter.extract_final_text(response) == ""


# ─── Slice 8: AnthropicToolAdapter — serialize_tool_defs + deserialize_tool_calls


class TestAnthropicSerializeToolDefs:
    """Slice 8: AnthropicToolAdapter.serialize_tool_defs."""

    def test_anthropic_serialize_tool_defs_callable(self):
        """Pass callable, assert {'name': ..., 'description': ..., 'input_schema': {...}}."""

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return "sunny"

        adapter = AnthropicToolAdapter()
        result = adapter.serialize_tool_defs([get_weather])

        assert result is not None
        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a city."
        assert "input_schema" in tool

    def test_anthropic_serialize_tool_defs_empty(self):
        """Empty tools list returns None."""
        adapter = AnthropicToolAdapter()
        result = adapter.serialize_tool_defs([])
        assert result is None


class TestAnthropicDeserializeToolCalls:
    """Slice 8: AnthropicToolAdapter.deserialize_tool_calls."""

    def test_anthropic_deserialize_tool_calls_from_response(self):
        """Construct ModelChatResponse with tool_calls, assert list[CanonicalToolCall]."""
        tc = _make_tool_call(id="toolu_01XYZ", name="search", arguments='{"q":"hello"}')
        response = _make_response(tool_calls=[tc])

        adapter = AnthropicToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 1
        call = result[0]
        assert isinstance(call, CanonicalToolCall)
        assert call.id == "toolu_01XYZ"
        assert call.name == "search"
        assert isinstance(call.arguments, dict)
        assert call.arguments["q"] == "hello"


# ─── Slice 9: AnthropicToolAdapter — serialize_tool_results + validate_pair_integrity


class TestAnthropicSerializeToolResults:
    """Slice 9: AnthropicToolAdapter.serialize_tool_results."""

    def test_anthropic_serialize_tool_results_bundles_in_one_user_message(self):
        """2 ToolResult objects, assert exactly ONE role='user' message appended with content=[tool_result blocks]."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        # Add an assistant message with tool_calls (required for completeness check)
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "toolu_1", "function": {"name": "tool_a"}},
                {"id": "toolu_2", "function": {"name": "tool_b"}},
            ]
        )

        results = [
            ToolResult(tool_call_id="toolu_1", name="tool_a", content="result_a"),
            ToolResult(tool_call_id="toolu_2", name="tool_b", content="result_b"),
        ]

        adapter = AnthropicToolAdapter()
        adapter.serialize_tool_results(results, chat)

        # Should have: user + assistant + 1 bundled user = 3 (no system message)
        assert len(chat.messages) == 3
        last_msg = chat.messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 2
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "toolu_1"
        assert last_msg["content"][1]["tool_use_id"] == "toolu_2"

    def test_anthropic_serialize_tool_results_raises_on_incomplete(self):
        """Assistant has 2 tool_use blocks but only 1 ToolResult provided, assert ValueError with 'missing'."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "toolu_1", "function": {"name": "tool_a"}},
                {"id": "toolu_2", "function": {"name": "tool_b"}},
            ]
        )

        # Only 1 result for 2 tool calls
        results = [
            ToolResult(tool_call_id="toolu_1", name="tool_a", content="result_a"),
        ]

        adapter = AnthropicToolAdapter()
        with pytest.raises(ValueError, match="missing"):
            adapter.serialize_tool_results(results, chat)


class TestAnthropicValidatePairIntegrity:
    """Slice 9: AnthropicToolAdapter.validate_pair_integrity."""

    def test_anthropic_validate_pair_integrity_complete_returns_true(self):
        """All tool_use IDs have matching results, assert True."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "toolu_1", "function": {"name": "tool_a"}},
                {"id": "toolu_2", "function": {"name": "tool_b"}},
            ]
        )
        chat.add_tool_result(tool_call_id="toolu_1", content="result_a")
        chat.add_tool_result(tool_call_id="toolu_2", content="result_b")

        adapter = AnthropicToolAdapter()
        assert adapter.validate_pair_integrity(chat) is True

    def test_anthropic_validate_pair_integrity_incomplete_returns_false(self):
        """Missing a tool result, assert False."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "toolu_1", "function": {"name": "tool_a"}},
                {"id": "toolu_2", "function": {"name": "tool_b"}},
            ]
        )
        chat.add_tool_result(tool_call_id="toolu_1", content="result_a")

        adapter = AnthropicToolAdapter()
        assert adapter.validate_pair_integrity(chat) is False


class TestAnthropicIsFinished:
    """AnthropicToolAdapter.is_finished."""

    def test_anthropic_is_finished_stop_returns_true(self):
        """finish_reason='stop' (mapped from end_turn), assert True."""
        response = _make_response(finish_reason="stop")
        adapter = AnthropicToolAdapter()
        assert adapter.is_finished(response) is True

    def test_anthropic_is_finished_tool_calls_returns_false(self):
        """finish_reason='tool_calls' (mapped from tool_use), assert False."""
        response = _make_response(finish_reason="tool_calls")
        adapter = AnthropicToolAdapter()
        assert adapter.is_finished(response) is False


class TestAnthropicExtractFinalText:
    """AnthropicToolAdapter.extract_final_text."""

    def test_anthropic_extract_final_text_returns_content(self):
        response = _make_response(content="Hello from Claude!")
        adapter = AnthropicToolAdapter()
        assert adapter.extract_final_text(response) == "Hello from Claude!"


# ─── Slice 10: ToolAdapterFactory ───────────────────────────────────────────


class TestToolAdapterFactory:
    """Slice 10: Factory auto-detect + fallback + registration."""

    def test_factory_create_openai_returns_openai_adapter(self):
        """create('openai') -> OpenAIToolAdapter."""
        adapter = ToolAdapterFactory.create("openai")
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_factory_create_anthropic_returns_anthropic_adapter(self):
        """create('anthropic') -> AnthropicToolAdapter."""
        adapter = ToolAdapterFactory.create("anthropic")
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_factory_create_unknown_falls_back_to_openai(self):
        """create('unknown_engine') -> OpenAIToolAdapter."""
        adapter = ToolAdapterFactory.create("unknown_engine")
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_factory_create_for_client_openai_engine(self):
        """Mock client with type(client.llm).engine == 'openai', assert OpenAIToolAdapter."""
        mock_engine_class = type("MockEngine", (), {"engine": "openai"})
        mock_engine = mock_engine_class()
        client = MagicMock()
        client.llm = mock_engine

        adapter = ToolAdapterFactory.create_for_client(client)
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_factory_create_for_client_anthropic_engine(self):
        """Mock client with type(client.llm).engine == 'anthropic', assert AnthropicToolAdapter."""
        mock_engine_class = type("MockEngine", (), {"engine": "anthropic"})
        mock_engine = mock_engine_class()
        client = MagicMock()
        client.llm = mock_engine

        adapter = ToolAdapterFactory.create_for_client(client)
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_factory_create_for_client_unknown_engine(self):
        """Mock client with no engine attr, assert OpenAIToolAdapter (fallback)."""
        client = MagicMock()
        client.llm = None

        adapter = ToolAdapterFactory.create_for_client(client)
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_factory_create_google_returns_gemini_adapter(self):
        """create('google') -> GeminiToolAdapter (native Gemini tooling)."""
        adapter = ToolAdapterFactory.create("google")
        assert isinstance(adapter, GeminiToolAdapter)

    def test_factory_amazon_falls_back_to_openai(self):
        """create('amazon') -> OpenAIToolAdapter (Bedrock OUT OF SCOPE)."""
        adapter = ToolAdapterFactory.create("amazon")
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_factory_openai_compatible_providers_use_openai_adapter(self):
        """All OpenAI-compatible providers return OpenAIToolAdapter."""
        providers = [
            "openrouter", "deepinfra", "groq", "together", "fireworks",
            "anyscale", "perplexity", "mistral", "cerebras", "friendliai",
            "novita", "deepseek", "sambanova", "azure", "cloudflare", "cohere",
        ]
        for provider in providers:
            adapter = ToolAdapterFactory.create(provider)
            assert isinstance(adapter, OpenAIToolAdapter), f"Failed for {provider}"


# ─── Slice 11: Adapter package re-exports ───────────────────────────────────


class TestAdapterReExports:
    """Slice 11: Adapter package re-exports."""

    def test_adapters_package_reexports_openai_adapter(self):
        """from magic_llm.agent.adapters import OpenAIToolAdapter works."""
        from magic_llm.agent.adapters import OpenAIToolAdapter as ReExported

        # Should be the same class as from source module
        assert ReExported is OpenAIToolAdapter

    def test_adapters_package_reexports_anthropic_adapter(self):
        """from magic_llm.agent.adapters import AnthropicToolAdapter works."""
        from magic_llm.agent.adapters import AnthropicToolAdapter as ReExported

        assert ReExported is AnthropicToolAdapter

    def test_adapters_package_reexports_gemini_adapter(self):
        """from magic_llm.agent.adapters import GeminiToolAdapter works."""
        from magic_llm.agent.adapters import GeminiToolAdapter as ReExported

        assert ReExported is GeminiToolAdapter
