"""Unit tests for GeminiToolAdapter serialization/deserialization.

Tests cover:
- Protocol conformance
- serialize_tool_defs: callable, dict, Pydantic model, empty list
- deserialize_tool_calls: single, multiple, none, malformed JSON
- serialize_tool_results: single, multiple, error, incomplete
- is_finished: with/without tool_calls, text + tool_calls
- extract_final_text: content present/absent
- validate_pair_integrity: matched, missing, no tool calls
"""

import json
from typing import Optional
from unittest.mock import MagicMock

import pytest

from pydantic import BaseModel

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, ToolCall, FunctionCall
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.agent.adapters import GeminiToolAdapter


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


# ─── Protocol conformance ──────────────────────────────────────────────────


class TestGeminiProtocolConformance:
    """GeminiToolAdapter satisfies ToolAdapter protocol."""

    def test_isinstance_tool_adapter(self):
        """isinstance(GeminiToolAdapter(), ToolAdapter) is True."""
        adapter = GeminiToolAdapter()
        assert isinstance(adapter, ToolAdapter)

    def test_has_all_required_methods(self):
        """All 6 methods present."""
        required_methods = [
            "serialize_tool_defs",
            "deserialize_tool_calls",
            "serialize_tool_results",
            "is_finished",
            "extract_final_text",
            "validate_pair_integrity",
        ]
        adapter = GeminiToolAdapter()
        for method_name in required_methods:
            assert hasattr(adapter, method_name), f"Missing {method_name}"


# ─── serialize_tool_defs ───────────────────────────────────────────────────


class TestGeminiSerializeToolDefs:
    """GeminiToolAdapter.serialize_tool_defs."""

    def test_callable_tool(self):
        """Callable tool → functionDeclarations with parametersJsonSchema + additionalProperties: false."""

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return "sunny"

        adapter = GeminiToolAdapter()
        result = adapter.serialize_tool_defs([get_weather])

        assert result is not None
        assert len(result) == 1
        assert "functionDeclarations" in result[0]

        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        tool = decls[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a city."
        assert "parametersJsonSchema" in tool

        schema = tool["parametersJsonSchema"]
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert schema["properties"]["city"]["type"] == "string"
        assert "city" in schema["required"]
        assert schema["additionalProperties"] is False

    def test_dict_tool_spec(self):
        """Dict spec → correct functionDeclarations format."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }

        adapter = GeminiToolAdapter()
        result = adapter.serialize_tool_defs([tool_spec])

        assert result is not None
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        tool = decls[0]
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web"
        assert "parametersJsonSchema" in tool
        assert tool["parametersJsonSchema"]["additionalProperties"] is False

    def test_pydantic_model_with_nested_defs(self):
        """Pydantic model with nested $defs → inlined schema, no $defs in output."""

        class Address(BaseModel):
            """Address details."""
            street: str
            city: str

        class Person(BaseModel):
            """A person with an address."""
            name: str
            address: Address

        adapter = GeminiToolAdapter()
        result = adapter.serialize_tool_defs([Person])

        assert result is not None
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        tool = decls[0]
        assert tool["name"] == "Person"

        schema = tool["parametersJsonSchema"]
        # No $defs or definitions at top level
        assert "$defs" not in schema
        assert "definitions" not in schema
        # Address should be inlined
        assert "address" in schema["properties"]
        address_schema = schema["properties"]["address"]
        assert address_schema.get("type") == "object"
        assert "street" in address_schema.get("properties", {})
        assert "city" in address_schema.get("properties", {})
        assert schema["additionalProperties"] is False

    def test_empty_list_returns_none(self):
        """Empty tools list returns None."""
        adapter = GeminiToolAdapter()
        result = adapter.serialize_tool_defs([])
        assert result is None


# ─── deserialize_tool_calls ────────────────────────────────────────────────


class TestGeminiDeserializeToolCalls:
    """GeminiToolAdapter.deserialize_tool_calls."""

    def test_single_tool_call(self):
        """Single tool call → [CanonicalToolCall] with parsed args dict."""
        tc = _make_tool_call(id="call_0_123", name="get_weather", arguments='{"city":"London"}')
        response = _make_response(tool_calls=[tc])

        adapter = GeminiToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 1
        call = result[0]
        assert isinstance(call, CanonicalToolCall)
        assert call.id == "call_0_123"
        assert call.name == "get_weather"
        assert isinstance(call.arguments, dict)
        assert call.arguments["city"] == "London"

    def test_multiple_parallel_tool_calls(self):
        """Multiple tool calls → list of CanonicalToolCall with distinct IDs."""
        tc1 = _make_tool_call(id="call_0_1", name="get_weather", arguments='{"city":"London"}')
        tc2 = _make_tool_call(id="call_1_2", name="search", arguments='{"q":"hello"}')
        response = _make_response(tool_calls=[tc1, tc2])

        adapter = GeminiToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 2
        assert result[0].id == "call_0_1"
        assert result[0].name == "get_weather"
        assert result[1].id == "call_1_2"
        assert result[1].name == "search"

    def test_no_tool_calls_returns_empty(self):
        """No tool_calls → empty list."""
        response = _make_response(content="Hello!")
        adapter = GeminiToolAdapter()
        result = adapter.deserialize_tool_calls(response)
        assert result == []

    def test_malformed_json_args_returns_empty_dict(self):
        """Malformed JSON in arguments → CanonicalToolCall with arguments = {}."""
        tc = _make_tool_call(id="call_1", name="bad_tool", arguments="not json")
        response = _make_response(tool_calls=[tc])

        adapter = GeminiToolAdapter()
        result = adapter.deserialize_tool_calls(response)

        assert len(result) == 1
        assert result[0].arguments == {}


# ─── serialize_tool_results ────────────────────────────────────────────────


class TestGeminiSerializeToolResults:
    """GeminiToolAdapter.serialize_tool_results."""

    def test_single_result(self):
        """Single result → one role="user" message with one functionResponse part, response: {"output": content}."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(
            tool_calls=[{"id": "call_0_123", "function": {"name": "get_weather"}}]
        )

        result = ToolResult(tool_call_id="call_0_123", name="get_weather", content="sunny")
        adapter = GeminiToolAdapter()
        adapter.serialize_tool_results([result], chat)

        # Should have: user + assistant + 1 bundled user = 3 (no system message)
        assert len(chat.messages) == 3
        last_msg = chat.messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 1

        part = last_msg["content"][0]
        assert "functionResponse" in part
        fr = part["functionResponse"]
        assert fr["name"] == "get_weather"
        assert fr["id"] == "call_0_123"
        assert fr["response"] == {"output": "sunny"}

    def test_two_results_bundled(self):
        """Two results → one role="user" message with two functionResponse parts."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool_a"}},
                {"id": "call_2", "function": {"name": "tool_b"}},
            ]
        )

        results = [
            ToolResult(tool_call_id="call_1", name="tool_a", content="result_a"),
            ToolResult(tool_call_id="call_2", name="tool_b", content="result_b"),
        ]

        adapter = GeminiToolAdapter()
        adapter.serialize_tool_results(results, chat)

        assert len(chat.messages) == 3
        last_msg = chat.messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert len(last_msg["content"]) == 2

        assert last_msg["content"][0]["functionResponse"]["id"] == "call_1"
        assert last_msg["content"][0]["functionResponse"]["response"] == {"output": "result_a"}
        assert last_msg["content"][1]["functionResponse"]["id"] == "call_2"
        assert last_msg["content"][1]["functionResponse"]["response"] == {"output": "result_b"}

    def test_error_result(self):
        """Error result → response: {"error": content}."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(
            tool_calls=[{"id": "call_1", "function": {"name": "broken_tool"}}]
        )

        result = ToolResult(
            tool_call_id="call_1", name="broken_tool", content="Error: timeout", is_error=True
        )
        adapter = GeminiToolAdapter()
        adapter.serialize_tool_results([result], chat)

        last_msg = chat.messages[-1]
        fr = last_msg["content"][0]["functionResponse"]
        assert fr["response"] == {"error": "Error: timeout"}

    def test_incomplete_results_raises_valueerror(self):
        """Incomplete results (2 tool_calls, 1 result) → ValueError with 'missing'."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool_a"}},
                {"id": "call_2", "function": {"name": "tool_b"}},
            ]
        )

        # Only 1 result for 2 tool calls
        results = [
            ToolResult(tool_call_id="call_1", name="tool_a", content="result_a"),
        ]

        adapter = GeminiToolAdapter()
        with pytest.raises(ValueError, match="missing"):
            adapter.serialize_tool_results(results, chat)


# ─── is_finished ───────────────────────────────────────────────────────────


class TestGeminiIsFinished:
    """GeminiToolAdapter.is_finished."""

    def test_no_tool_calls_returns_true(self):
        """tool_calls=None → True."""
        response = _make_response(content="Hello!", tool_calls=None, finish_reason="stop")
        adapter = GeminiToolAdapter()
        assert adapter.is_finished(response) is True

    def test_with_tool_calls_returns_false(self):
        """tool_calls=[...] → False."""
        tc = _make_tool_call()
        response = _make_response(tool_calls=[tc], finish_reason="stop")
        adapter = GeminiToolAdapter()
        assert adapter.is_finished(response) is False

    def test_text_and_tool_calls_returns_false(self):
        """text + tool_calls → False (tool calls take priority)."""
        tc = _make_tool_call()
        response = _make_response(content="Let me check...", tool_calls=[tc], finish_reason="stop")
        adapter = GeminiToolAdapter()
        assert adapter.is_finished(response) is False


# ─── extract_final_text ────────────────────────────────────────────────────


class TestGeminiExtractFinalText:
    """GeminiToolAdapter.extract_final_text."""

    def test_content_present(self):
        """Content present → returns content."""
        response = _make_response(content="The weather is sunny.")
        adapter = GeminiToolAdapter()
        assert adapter.extract_final_text(response) == "The weather is sunny."

    def test_content_none(self):
        """Content None → returns empty string."""
        response = _make_response(content=None)
        adapter = GeminiToolAdapter()
        assert adapter.extract_final_text(response) == ""


# ─── validate_pair_integrity ───────────────────────────────────────────────


class TestGeminiValidatePairIntegrity:
    """GeminiToolAdapter.validate_pair_integrity."""

    def test_all_ids_matched(self):
        """All functionResponse IDs matched → True."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool_a"}},
                {"id": "call_2", "function": {"name": "tool_b"}},
            ]
        )
        chat.add_tool_messages([
            {
                "role": "user",
                "content": [
                    {"functionResponse": {"id": "call_1", "name": "tool_a", "response": {"output": "a"}}},
                    {"functionResponse": {"id": "call_2", "name": "tool_b", "response": {"output": "b"}}},
                ],
            }
        ])

        adapter = GeminiToolAdapter()
        assert adapter.validate_pair_integrity(chat) is True

    def test_missing_result(self):
        """Missing a functionResponse → False."""
        chat = ModelChat()
        chat.add_tool_call_message(
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool_a"}},
                {"id": "call_2", "function": {"name": "tool_b"}},
            ]
        )
        chat.add_tool_messages([
            {
                "role": "user",
                "content": [
                    {"functionResponse": {"id": "call_1", "name": "tool_a", "response": {"output": "a"}}},
                ],
            }
        ])

        adapter = GeminiToolAdapter()
        assert adapter.validate_pair_integrity(chat) is False

    def test_no_tool_calls_in_history(self):
        """No tool calls in history → True."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_assistant_message("Hi there!")

        adapter = GeminiToolAdapter()
        assert adapter.validate_pair_integrity(chat) is True
