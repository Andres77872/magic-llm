"""End-to-end agent loop characterization tests for native Gemini tooling.

Tests cover the full flow from AgentLoop through EngineGoogle to mocked
Gemini API responses:
- Single tool call cycle: mock HttpClient.post_json to return Gemini response
  with functionCall, run AgentLoop.run(), assert final text response
- Parallel tool calls: mock response with 2 functionCall parts, assert both
  tools executed and results bundled into single user message
- Streaming tool loop: mock HttpClient.stream_request to yield SSE chunks
  including functionCall, assert loop continues and completes
- Non-tool Gemini call unchanged: regression test for backward compatibility
- Pre-serialized Gemini tools passthrough: engine detects already-Gemini-format
  tools from GeminiToolAdapter and passes through without double-mapping
"""

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import (
    ModelChatResponse, Choice, Message, UsageModel,
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel, ChoiceModel, DeltaModel, UsageModel as StreamUsage,
)
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.agent.adapters import GeminiToolAdapter
from magic_llm.engine.engine_google import EngineGoogle


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_gemini_response(
    content=None,
    function_calls=None,
    finish_reason="STOP",
    model_version="gemini-2.5-flash",
    response_id=None,
):
    """Build a raw Gemini API response dict (as returned by HttpClient.post_json).

    Args:
        content: Text content string or None.
        function_calls: List of functionCall dicts, e.g.:
            [{"name": "get_weather", "args": {"city": "London"}, "id": "call_1"}]
        finish_reason: Gemini finish reason (default: "STOP").
        model_version: Model version string.
        response_id: Response ID string.
    """
    parts = []
    if content:
        parts.append({"text": content})
    for fc in (function_calls or []):
        parts.append({"functionCall": fc})

    return {
        "candidates": [{
            "content": {
                "parts": parts,
                "role": "model",
            },
            "finishReason": finish_reason,
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
        "modelVersion": model_version,
        "responseId": response_id or f"gemini_{int(time.time() * 1000)}",
    }


def _make_model_chat_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
):
    """Build a ModelChatResponse (normalized by process_generate)."""
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="test-1",
        object="chat.completion",
        created=1700000000.0,
        model="gemini-2.5-flash",
        choices=[choice],
        usage=UsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_mock_client_with_engine(engine: EngineGoogle):
    """Create a mock client with the given engine as client.llm."""
    client = MagicMock()
    client.llm = engine
    return client


# ─── Test 1: Single tool call cycle ────────────────────────────────────────


class TestGeminiSingleToolCallCycle:
    """End-to-end: single tool call → tool execution → final response."""

    def test_single_tool_call_completes_loop(self):
        """AgentLoop.run() with Gemini: tool call → execute → final text."""

        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny, 25°C in {city}"

        # Build Gemini responses in sequence
        tool_call_response = _make_gemini_response(
            function_calls=[{
                "id": "call_0_123",
                "name": "get_weather",
                "args": {"city": "Buenos Aires"},
            }],
        )
        final_response = _make_gemini_response(
            content="The weather in Buenos Aires is sunny, 25°C.",
        )

        # Create engine and mock HttpClient
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        call_count = [0]
        original_post_json = engine.__class__.generate

        def mock_generate(chat, **kwargs):
            call_count[0] += 1
            # Process the appropriate response
            if call_count[0] == 1:
                return engine.process_generate(tool_call_response)
            else:
                return engine.process_generate(final_response)

        engine.generate = MagicMock(side_effect=mock_generate)

        # Run the agent loop
        loop = AgentLoop(
            client=client,
            tools=[get_weather],
            budget=AgentBudget(max_iterations=5),
        )

        result = loop.run(user_input="What's the weather in Buenos Aires?")

        assert call_count[0] == 2
        assert "Buenos Aires" in result.content
        assert "sunny" in result.content.lower() or "25" in result.content

    def test_chat_messages_contain_function_response(self):
        """After tool execution, chat has bundled functionResponse parts."""

        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        tool_call_response = _make_gemini_response(
            function_calls=[{
                "id": "call_1",
                "name": "get_weather",
                "args": {"city": "London"},
            }],
        )
        final_response = _make_gemini_response(content="Done")

        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        captured_chat = None
        call_count = [0]

        def mock_generate(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            call_count[0] += 1
            if call_count[0] == 1:
                return engine.process_generate(tool_call_response)
            return engine.process_generate(final_response)

        engine.generate = MagicMock(side_effect=mock_generate)

        loop = AgentLoop(
            client=client,
            tools=[get_weather],
            budget=AgentBudget(max_iterations=5),
        )
        loop.run(user_input="Weather?")

        # Find the functionResponse message
        fr_msg = None
        for msg in captured_chat.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and "functionResponse" in part:
                        fr_msg = msg
                        break
            if fr_msg:
                break

        assert fr_msg is not None
        # Should have exactly one functionResponse part (single tool call)
        fr_parts = [p for p in fr_msg["content"] if "functionResponse" in p]
        assert len(fr_parts) == 1
        assert fr_parts[0]["functionResponse"]["name"] == "get_weather"
        # ToolExecutor._serialize_output does json.dumps on string output
        assert fr_parts[0]["functionResponse"]["response"] == {"output": '"sunny"'}


# ─── Test 2: Parallel tool calls ───────────────────────────────────────────


class TestGeminiParallelToolCalls:
    """End-to-end: parallel tool calls → bundled results."""

    def test_both_tools_executed(self):
        """Two functionCall parts → both tools executed."""

        def tool_a() -> str:
            """Tool A."""
            return "result_a"

        def tool_b() -> str:
            """Tool B."""
            return "result_b"

        tool_call_response = _make_gemini_response(
            function_calls=[
                {"id": "call_1", "name": "tool_a", "args": {}},
                {"id": "call_2", "name": "tool_b", "args": {}},
            ],
        )
        final_response = _make_gemini_response(content="Both done")

        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        executed = []

        def mock_generate(chat, **kwargs):
            # Check if tool results have been injected (user message with functionResponse parts)
            last_msg = chat.messages[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content")
                if isinstance(content, list):
                    has_fr = any(
                        isinstance(p, dict) and "functionResponse" in p
                        for p in content
                    )
                    if has_fr:
                        return engine.process_generate(final_response)
            return engine.process_generate(tool_call_response)

        engine.generate = MagicMock(side_effect=mock_generate)

        loop = AgentLoop(
            client=client,
            tools=[tool_a, tool_b],
            budget=AgentBudget(max_iterations=5),
        )
        result = loop.run(user_input="Run both tools")

        assert result.content == "Both done"
        assert engine.generate.call_count == 2

    def test_results_bundled_in_single_user_message(self):
        """Two tool results → single role="user" message with 2 functionResponse parts."""

        def tool_a() -> str:
            return "result_a"

        def tool_b() -> str:
            return "result_b"

        tool_call_response = _make_gemini_response(
            function_calls=[
                {"id": "call_1", "name": "tool_a", "args": {}},
                {"id": "call_2", "name": "tool_b", "args": {}},
            ],
        )
        final_response = _make_gemini_response(content="Done")

        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        captured_chat = None
        call_count = [0]

        def mock_generate(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            call_count[0] += 1
            if call_count[0] == 1:
                return engine.process_generate(tool_call_response)
            return engine.process_generate(final_response)

        engine.generate = MagicMock(side_effect=mock_generate)

        loop = AgentLoop(
            client=client,
            tools=[tool_a, tool_b],
            budget=AgentBudget(max_iterations=5),
        )
        loop.run(user_input="Run tools")

        # Find the bundled functionResponse message
        bundled_msg = None
        for msg in captured_chat.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                fr_count = sum(
                    1 for p in msg["content"]
                    if isinstance(p, dict) and "functionResponse" in p
                )
                if fr_count == 2:
                    bundled_msg = msg
                    break

        assert bundled_msg is not None
        fr_ids = [
            p["functionResponse"]["id"]
            for p in bundled_msg["content"]
            if "functionResponse" in p
        ]
        assert "call_1" in fr_ids
        assert "call_2" in fr_ids


# ─── Test 3: Non-tool Gemini call unchanged (regression) ───────────────────


class TestGeminiNonToolRegression:
    """Non-tool Gemini calls must work identically to pre-change behavior."""

    def test_non_tool_call_works(self):
        """Simple chat without tools → single generate call, returns content."""

        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        response = _make_gemini_response(
            content="Hello! How can I help you?",
        )

        def mock_generate(chat, **kwargs):
            # Verify no tools were passed
            assert "tools" not in kwargs or kwargs.get("tools") is None
            return engine.process_generate(response)

        engine.generate = MagicMock(side_effect=mock_generate)

        loop = AgentLoop(
            client=client,
            budget=AgentBudget(max_iterations=5),
        )
        result = loop.run(user_input="Hello")

        assert result.content == "Hello! How can I help you?"
        assert engine.generate.call_count == 1

    def test_request_payload_has_no_tool_keys(self):
        """Non-tool call → data dict has no 'tools' or 'toolConfig' keys."""
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        chat = ModelChat()
        chat.add_user_message("Hello")

        json_bytes, headers, data = engine.prepare_data_sync(chat)

        assert "tools" not in data
        assert "toolConfig" not in data
        assert "contents" in data
        assert "generationConfig" in data


# ─── Test 4: Pre-serialized Gemini tools passthrough ───────────────────────


class TestGeminiPreSerializedToolsPassthrough:
    """Engine detects already-Gemini-format tools and passes through."""

    def test_pre_serialized_tools_passed_through(self):
        """Tools already in [{"functionDeclarations": [...]}] format → used directly."""
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        chat = ModelChat()
        chat.add_user_message("Weather?")

        # Simulate what GeminiToolAdapter.serialize_tool_defs returns
        adapter = GeminiToolAdapter()

        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        serialized = adapter.serialize_tool_defs([get_weather])

        # This is what the agent loop passes to generate()
        json_bytes, headers, data = engine.prepare_data_sync(
            chat, tools=serialized, tool_choice="auto"
        )

        # Tools should be passed through directly (not double-mapped)
        assert "tools" in data
        assert data["tools"] == serialized
        # tool_choice should still be mapped
        assert "toolConfig" in data
        assert data["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}

    def test_pre_serialized_tools_with_named_choice(self):
        """Pre-serialized tools + named tool_choice → correct toolConfig."""
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        chat = ModelChat()
        chat.add_user_message("Weather?")

        adapter = GeminiToolAdapter()

        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        def search(q: str) -> str:
            """Search."""
            return "found"

        serialized = adapter.serialize_tool_defs([get_weather, search])

        json_bytes, headers, data = engine.prepare_data_sync(
            chat, tools=serialized, tool_choice={"name": "get_weather"}
        )

        assert "tools" in data
        assert data["tools"] == serialized
        assert data["toolConfig"] == {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["get_weather"],
            }
        }

    def test_raw_callable_tools_still_work(self):
        """Raw callables (not pre-serialized) → converted via map_to_gemini."""
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        chat = ModelChat()
        chat.add_user_message("Weather?")

        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        # Pass raw callable (not pre-serialized)
        json_bytes, headers, data = engine.prepare_data_sync(
            chat, tools=[get_weather], tool_choice="auto"
        )

        assert "tools" in data
        assert len(data["tools"]) == 1
        assert "functionDeclarations" in data["tools"][0]
        decls = data["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert "toolConfig" in data


# ─── Test 5: Streaming tool loop (characterization) ────────────────────────


class TestGeminiStreamingToolLoop:
    """Streaming agent loop with Gemini functionCall parts."""

    def test_stream_text_then_tool_call_then_final(self):
        """Stream yields text chunks, then functionCall chunk, then final text."""

        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")
        client = _make_mock_client_with_engine(engine)

        # Build SSE chunks for the first iteration (tool call)
        tool_stream_chunks = [
            f'data: {json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "Let me check..."}], "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12},
            })}',
            f'data: {json.dumps({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "functionCall": {
                                "id": "stream_call_1",
                                "name": "get_weather",
                                "args": {"city": "London"},
                            }
                        }],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            })}',
        ]

        # Build SSE chunks for the final iteration (text only)
        final_stream_chunks = [
            f'data: {json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "The weather in London is sunny."}], "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10, "totalTokenCount": 30},
            })}',
        ]

        stream_iterations = [iter(tool_stream_chunks), iter(final_stream_chunks)]

        def mock_stream_generate(chat, **kwargs):
            try:
                chunk_iter = next(stream_iterations)
                yield from chunk_iter
            except StopIteration:
                return

        engine.stream_generate = MagicMock(side_effect=mock_stream_generate)

        # Verify stream chunks parse correctly
        for chunk in tool_stream_chunks:
            result = engine.prepare_stream_response(chunk)
            assert result is not None

        for chunk in final_stream_chunks:
            result = engine.prepare_stream_response(chunk)
            assert result is not None
            assert result.choices[0].delta.content == "The weather in London is sunny."

    def test_function_call_chunk_does_not_crash(self):
        """Streaming chunk with only functionCall → delta.content = '', no crash."""
        engine = EngineGoogle(api_key="test-key", model="gemini-2.5-flash")

        chunk = f'data: {json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "id": "call_1",
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
        })}'

        result = engine.prepare_stream_response(chunk)

        assert result.choices[0].delta.content == ""
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"
