"""Integration-style tests for the agentic loop — realistic provider-facing shapes.

TDD: agentic-tooling-flow-tests — Phase 5 (Integration)

These tests exercise the agentic loop with realistic provider response shapes,
message history assertions, and adapter expectations. They do NOT require live
API keys — all provider interactions are mocked but use realistic data structures
that mirror actual OpenAI/Anthropic response formats.

Purpose:
1. Bridge the gap between pure unit mocks and live integration tests
2. Exercise realistic ModelChatResponse / stream chunk shapes
3. Assert on message/history structure showing current vs adapter expectations
4. Cover both non-streaming and streaming integration-like paths

Slices:
- Slice V: Extend e2e with message assertions (integration-style, no live keys)
- Slice W: Streaming integration-style (realistic chunk shapes)
- Additional: Provider-facing shape tests, tool_choice variations, dict tool specs
"""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import (
    run_agentic,
    run_agentic_stream,
    _format_tool_feedback,
)
from magic_llm.model.ModelChat import ModelChat
from magic_llm.model.ModelChatResponse import (
    ModelChatResponse, Choice, Message, UsageModel,
    ToolCall, FunctionCall,
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel, ChoiceModel, DeltaModel, UsageModel as StreamUsageModel,
)
from magic_llm.agent.types import ToolResult, CanonicalToolCall


# ─── Helper builders ──────────────────────────────────────────────────────

def _make_mock_client(responses: list):
    """Create a mock client that returns responses in sequence."""
    client = MagicMock()
    client.llm.generate = MagicMock(side_effect=responses)
    return client


def _make_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
    model="gpt-4-0125-preview",
    usage=None,
):
    """Build a valid ModelChatResponse with realistic provider shapes."""
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="chatcmpl-9XyZ1234567890abcdef",
        object="chat.completion",
        created=1700000000,
        model=model,
        choices=[choice],
        usage=usage or UsageModel(
            prompt_tokens=45, completion_tokens=12, total_tokens=57,
        ),
        system_fingerprint="fp_abc123",
    )


def _make_tool_call(id="call_abc123", name="get_weather", arguments='{"city":"London"}'):
    """Build a valid ToolCall with realistic provider ID format."""
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))


def _make_stream_chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    model="gpt-4-0125-preview",
    chunk_id="chatcmpl-chunk-1",
):
    """Build a realistic ChatCompletionModel stream chunk."""
    delta = DeltaModel(content=content, tool_calls=tool_calls)
    choice = ChoiceModel(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionModel(
        id=chunk_id,
        model=model,
        choices=[choice],
        usage=StreamUsageModel(),
    )


def _make_stream_tool_call_delta(
    index=0,
    call_id=None,
    name=None,
    arguments=None,
):
    """Build a streaming tool call delta (as received from provider)."""
    from magic_llm.model.ModelChatStream import ToolCall as StreamToolCall, FunctionCall as StreamFunctionCall
    return StreamToolCall(
        index=index,
        id=call_id,
        function=StreamFunctionCall(name=name, arguments=arguments),
    )


# ─── Slice V: Integration-style message assertions ────────────────────────

class TestAgenticMessageHistoryIntegration:
    """Integration-style tests asserting on chat.messages structure.

    These exercise the full run_agentic() loop with realistic provider shapes
    and assert on the resulting message history. They document both:
    - Current behavior (role:user with formatted text)
    - Adapter expectations (role:tool with tool_call_id)
    """

    def test_full_loop_message_history_no_tools(self):
        """Single-iteration loop produces user→assistant message history."""
        response = _make_response(content="The weather is sunny.", finish_reason="stop")
        client = _make_mock_client([response])

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "What's the weather in London?", tools=[])

        assert captured_chat is not None
        messages = captured_chat.messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What's the weather in London?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "The weather is sunny."

    def test_full_loop_message_history_single_tool_call(self):
        """Multi-turn loop: user→assistant(tool)→user(feedback)→assistant(final)."""
        tool_call = _make_tool_call(id="call_weather", name="get_weather",
                                    arguments='{"city":"London"}')
        tool_response = _make_response(content="Let me check...", tool_calls=[tool_call])
        final_response = _make_response(content="It's 18°C in London.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temperature": 18, "unit": "C", "description": "Partly cloudy"}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            # Detect tool feedback to return final response
            for msg in captured_chat.messages:
                if msg["role"] == "user" and "tool: get_weather" in str(msg.get("content", "")):
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(client, "Weather in London?", tools=[get_weather])

        assert captured_chat is not None
        messages = captured_chat.messages

        # Current behavior: user → assistant(content) → user(formatted_feedback) → assistant(final)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Weather in London?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Let me check..."
        # CURRENT: tool feedback as role:user (will change with SDD refactor)
        assert messages[2]["role"] == "user"
        assert "tool: get_weather" in messages[2]["content"]
        assert "London" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert "18" in result.content

    def test_full_loop_message_history_multi_tool_calls(self):
        """Multi-tool loop: all results combined into single user message."""
        call_a = _make_tool_call(id="call_a", name="search_papers",
                                 arguments='{"query":"LLMs"}')
        call_b = _make_tool_call(id="call_b", name="summarize",
                                 arguments='{"text":"..."}')
        tool_response = _make_response(content=None, tool_calls=[call_a, call_b])
        final_response = _make_response(content="Found 3 relevant papers.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def search_papers(query):
            return [{"title": "Paper 1"}, {"title": "Paper 2"}]

        def summarize(text):
            return "Summary of text"

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                content = str(msg.get("content", ""))
                if "tool: search_papers" in content and "tool: summarize" in content:
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Search and summarize papers about LLMs",
                    tools=[search_papers, summarize])

        messages = captured_chat.messages
        # user → user(combined_feedback) → assistant
        # Note: no assistant message because tool_response had content=None
        assert len(messages) == 3
        combined = messages[1]["content"]
        assert "tool: search_papers" in combined
        assert "tool: summarize" in combined
        # Results are separated by \n\n (current behavior)
        assert "\n\n" in combined

    def test_message_history_shows_adapter_expectation_gap(self):
        """Documents the gap between current behavior and what adapters expect.

        Current: role:user with _format_tool_feedback text
        Target:  role:tool with tool_call_id correlation

        This test is a PASSING characterization test that explicitly documents
        what adapters (OpenAIToolAdapter, AnthropicToolAdapter) will need to
        work with after the SDD refactor.
        """
        tool_call = _make_tool_call(id="call_abc123", name="get_weather",
                                    arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                if "tool: get_weather" in str(msg.get("content", "")):
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # Document current behavior
        tool_feedback_msgs = [
            m for m in captured_chat.messages
            if m["role"] == "user" and "tool: get_weather" in str(m.get("content", ""))
        ]
        assert len(tool_feedback_msgs) == 1
        feedback = tool_feedback_msgs[0]["content"]

        # Current format: _format_tool_feedback style
        expected_format = _format_tool_feedback(
            name="get_weather",
            input_str='{"city": "London"}',
            output_str='{"temp": 18}',
        )
        assert expected_format in feedback

        # Document what adapters EXPECT (this is NOT current behavior):
        # After refactor, there should be role:tool messages with tool_call_id
        tool_role_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_role_msgs) == 0, (
            "CURRENT: No role:tool messages exist yet. "
            "After SDD refactor, adapters expect role:tool with tool_call_id."
        )

    def test_message_history_with_extra_messages_seed(self):
        """Loop correctly handles extra_messages seeded into the conversation."""
        response = _make_response(content="Continuing from context.", finish_reason="stop")
        client = _make_mock_client([response])

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(
            client,
            user_input="What do you think?",
            extra_messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Previous context."},
                {"role": "assistant", "content": "I understand."},
            ],
        )

        assert captured_chat is not None
        messages = captured_chat.messages
        # system + extra user + extra assistant + new user = 4 before generate
        assert len(messages) >= 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Previous context."
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "I understand."

    def test_message_history_with_system_prompt(self):
        """System prompt is correctly placed as first message."""
        response = _make_response(content="Hello!", finish_reason="stop")
        client = _make_mock_client([response])

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(
            client,
            user_input="Hi",
            system_prompt="You are a weather assistant.",
        )

        assert captured_chat.messages[0]["role"] == "system"
        assert captured_chat.messages[0]["content"] == "You are a weather assistant."


class TestAgenticToolRegistryIntegration:
    """Integration tests for tool registry with realistic tool definitions.

    Tests dict tool specs, callable tools, and tool_functions override patterns
    that mirror real-world usage.
    """

    def test_callable_tool_registered_by_name(self):
        """Callable tools are registered by their __name__ attribute."""
        tool_call = _make_tool_call(name="calculate_sum", arguments='{"a": 5, "b": 3}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Result: 8", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def calculate_sum(a, b):
            return a + b

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                if "tool: calculate_sum" in str(msg.get("content", "")):
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(client, "Calculate 5 + 3", tools=[calculate_sum])

        assert "8" in result.content

    def test_tool_functions_override_callable_with_same_name(self):
        """tool_functions dict entries override callable entries with the same name."""
        tool_call = _make_tool_call(name="get_data", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_data():
            return "from_callable"

        def get_data_override():
            return "from_tool_functions"

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                content = str(msg.get("content", ""))
                if "tool: get_data" in content:
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        # Register callable, but override with tool_functions
        result = run_agentic(
            client, "test",
            tools=[get_data],
            tool_functions={"get_data": get_data_override},
        )

        # The override should have been used
        assert "from_tool_functions" in str(captured_chat.messages)
        assert "from_callable" not in str(captured_chat.messages)

    def test_dict_tool_spec_with_callable_execution(self):
        """Dict tool specs (OpenAI format) work when callable is provided via tool_functions."""
        # Realistic OpenAI-style tool definition
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database for relevant records",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        }

        tool_call = _make_tool_call(
            name="search_database",
            arguments='{"query":"climate change","limit":5}',
        )
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Found 5 results.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def search_database(query, limit=10):
            return {"results": [{"title": f"Result about {query}"}], "count": limit}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                if "tool: search_database" in str(msg.get("content", "")):
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(
            client, "Search for climate change papers",
            tools=[tool_spec],
            tool_functions={"search_database": search_database},
        )

        assert "5" in result.content


class TestAgenticToolChoiceIntegration:
    """Integration tests for tool_choice variations."""

    def test_tool_choice_auto_allows_tool_calls(self):
        """tool_choice='auto' allows the model to decide on tool calls."""
        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        captured_kwargs = []

        def capture_chat(chat, **kwargs):
            captured_kwargs.append(kwargs)
            for msg in chat.messages:
                if "tool: get_weather" in str(msg.get("content", "")):
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        def get_weather(city):
            return {"temp": 18}

        run_agentic(client, "Weather?", tools=[get_weather], tool_choice="auto")

        # tool_choice should be passed through to generate
        assert captured_kwargs[0].get("tool_choice") == "auto"

    def test_tool_choice_none_prevents_tool_calls(self):
        """tool_choice='none' is passed through to generate (model may still return tools)."""
        response = _make_response(content="I cannot use tools.", finish_reason="stop")
        client = _make_mock_client([response])

        captured_kwargs = []

        def capture_chat(chat, **kwargs):
            captured_kwargs.append(kwargs)
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        def dummy_tool():
            return "ok"

        run_agentic(client, "Hello", tools=[dummy_tool], tool_choice="none")

        assert captured_kwargs[0].get("tool_choice") == "none"

    def test_tool_choice_dict_passed_through(self):
        """tool_choice as dict (e.g., {'type': 'function', 'function': {'name': 'x'}}) works."""
        response = _make_response(content="Using forced tool.", finish_reason="stop")
        client = _make_mock_client([response])

        captured_kwargs = []

        def capture_chat(chat, **kwargs):
            captured_kwargs.append(kwargs)
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        def forced_tool():
            return "forced"

        run_agentic(
            client, "test",
            tools=[forced_tool],
            tool_choice={"type": "function", "function": {"name": "forced_tool"}},
        )

        assert captured_kwargs[0].get("tool_choice") == {
            "type": "function", "function": {"name": "forced_tool"}
        }


class TestAgenticFinishReasonIntegration:
    """Integration tests for finish_reason handling in realistic scenarios."""

    def test_finish_reason_stop_ends_loop(self):
        """finish_reason='stop' ends the loop normally."""
        response = _make_response(content="Final answer.", finish_reason="stop")
        client = _make_mock_client([response])

        result = run_agentic(client, "What is 2+2?")

        assert result.content == "Final answer."
        assert result.finish_reason == "stop"
        assert client.llm.generate.call_count == 1

    def test_finish_reason_length_ends_loop(self):
        """finish_reason='length' also ends the loop (no tool calls present)."""
        response = _make_response(content="Truncated response...", finish_reason="length")
        client = _make_mock_client([response])

        result = run_agentic(client, "Write a long essay")

        assert result.content == "Truncated response..."
        assert result.finish_reason == "length"
        assert client.llm.generate.call_count == 1

    def test_finish_reason_tool_calls_with_empty_list_breaks_loop_current(self):
        """CURRENT: finish_reason='tool_calls' with empty tool_calls list breaks the loop.

        This documents current behavior: the loop only checks tool_calls list,
        not finish_reason. After refactor, adapter.is_finished() should handle this.
        """
        # Edge case: model signals tool_calls needed but list is empty
        response = _make_response(
            content=None,
            tool_calls=[],  # Empty list
            finish_reason="tool_calls",
        )
        client = _make_mock_client([response])

        result = run_agentic(client, "test", tools=[], max_iterations=5)

        # CURRENT: Loop breaks because tool_calls list is empty
        assert client.llm.generate.call_count == 1
        assert result.content is None or result.content == ""

    def test_multi_iteration_finish_reason_preserved(self):
        """Final response's finish_reason is preserved in the returned response."""
        tool_call = _make_tool_call(name="fn", arguments='{}')
        responses = [
            _make_response(content="thinking", tool_calls=[tool_call], finish_reason="tool_calls"),
            _make_response(content="final answer", finish_reason="stop"),
        ]
        client = _make_mock_client(responses)

        def fn():
            return "ok"

        result = run_agentic(client, "test", tools=[fn])

        assert result.finish_reason == "stop"


# ─── Slice W: Streaming integration-style tests ───────────────────────────

class TestAgenticStreamIntegration:
    """Integration-style streaming tests with realistic chunk shapes.

    These exercise run_agentic_stream() with realistic provider chunk sequences,
    asserting on the full streaming cycle including tool execution and continuation.
    """

    def test_streaming_full_cycle_no_tools(self):
        """Full streaming cycle: all chunks yielded, no tool execution."""
        chunks = [
            _make_stream_chunk(content="The ", chunk_id="chunk-1"),
            _make_stream_chunk(content="weather ", chunk_id="chunk-2"),
            _make_stream_chunk(content="is sunny.", chunk_id="chunk-3", finish_reason="stop"),
        ]

        client = MagicMock()
        client.llm.stream_generate = MagicMock(return_value=iter(chunks))

        result_chunks = list(run_agentic_stream(client, "What's the weather?"))

        assert len(result_chunks) == 3
        assert result_chunks[0].choices[0].delta.content == "The "
        assert result_chunks[1].choices[0].delta.content == "weather "
        assert result_chunks[2].choices[0].delta.content == "is sunny."
        assert result_chunks[2].choices[0].finish_reason == "stop"
        assert client.llm.stream_generate.call_count == 1

    def test_streaming_full_cycle_with_tool_call(self):
        """Full streaming cycle: chunks → tool execution → separator → second iteration chunks."""
        # First iteration: text then tool call delta
        first_chunks = [
            _make_stream_chunk(content="Let me check...", chunk_id="chunk-1"),
            _make_stream_chunk(
                tool_calls=[_make_stream_tool_call_delta(
                    index=0, call_id="call_weather", name="get_weather",
                    arguments='{"city":"London"}',
                )],
                chunk_id="chunk-2",
                finish_reason="tool_calls",
            ),
        ]
        # Second iteration: final response
        second_chunks = [
            _make_stream_chunk(content="It's 18°C.", chunk_id="chunk-3", finish_reason="stop"),
        ]

        call_count = [0]

        def stream_generate(chat, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(first_chunks)
            return iter(second_chunks)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=stream_generate)

        def get_weather(city):
            return {"temp": 18}

        result_chunks = list(run_agentic_stream(client, "Weather in London?", tools=[get_weather]))

        # Should have: first_chunks + separator + second_chunks
        # Separator is only yielded when iteration_content is truthy
        assert len(result_chunks) >= 3  # At minimum: 2 first + 1 second
        assert call_count[0] == 2  # Two stream_generate calls

    def test_streaming_message_history_after_tool_execution(self):
        """Streaming path message history matches non-streaming path."""
        first_chunks = [
            _make_stream_chunk(content="Checking...", chunk_id="chunk-1"),
            _make_stream_chunk(
                tool_calls=[_make_stream_tool_call_delta(
                    index=0, call_id="call_1", name="lookup", arguments='{}',
                )],
                chunk_id="chunk-2",
            ),
        ]
        second_chunks = [
            _make_stream_chunk(content="Done.", chunk_id="chunk-3", finish_reason="stop"),
        ]

        call_count = [0]
        captured_chat = None

        def stream_generate(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(first_chunks)
            return iter(second_chunks)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=stream_generate)

        def lookup():
            return "found"

        list(run_agentic_stream(client, "test", tools=[lookup]))

        assert captured_chat is not None
        messages = captured_chat.messages

        # user → assistant(iteration_content) → user(feedback) → assistant(final)
        # Note: streaming adds assistant message with iteration_content
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        assert len(user_msgs) >= 2  # Original + tool feedback
        assert len(assistant_msgs) >= 1  # Iteration content

    def test_streaming_separator_only_when_iteration_content_truthy(self):
        """Separator chunk is only emitted when iteration has text content.

        This documents an important implementation detail: if a streaming
        iteration produces ONLY tool_call deltas with no text content,
        no separator is yielded between iterations.
        """
        # First iteration: ONLY tool call delta, no text content
        first_chunks = [
            _make_stream_chunk(
                tool_calls=[_make_stream_tool_call_delta(
                    index=0, call_id="call_1", name="fn", arguments='{}',
                )],
                chunk_id="chunk-1",
            ),
        ]
        second_chunks = [
            _make_stream_chunk(content="Result", chunk_id="chunk-2", finish_reason="stop"),
        ]

        call_count = [0]
        all_chunks = []

        def stream_generate(chat, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(first_chunks)
            return iter(second_chunks)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=stream_generate)

        def fn():
            return "ok"

        all_chunks = list(run_agentic_stream(client, "test", tools=[fn], content_separator="---"))

        # No separator should be present because first iteration had no text content
        separator_chunks = [
            c for c in all_chunks
            if c.choices[0].delta.content == "---"
        ]
        assert len(separator_chunks) == 0, (
            "Separator should NOT be emitted when iteration has no text content. "
            "Current behavior: separator only when iteration_content is truthy."
        )

    def test_streaming_multi_tool_accumulation_and_execution(self):
        """Streaming: multiple tool calls accumulated and executed sequentially."""
        first_chunks = [
            _make_stream_chunk(content="I'll use both tools...", chunk_id="chunk-1"),
            _make_stream_chunk(
                tool_calls=[
                    _make_stream_tool_call_delta(index=0, call_id="call_a", name="tool_a", arguments='{}'),
                    _make_stream_tool_call_delta(index=1, call_id="call_b", name="tool_b", arguments='{}'),
                ],
                chunk_id="chunk-2",
            ),
        ]
        second_chunks = [
            _make_stream_chunk(content="Both tools completed.", chunk_id="chunk-3", finish_reason="stop"),
        ]

        call_count = [0]
        executed = []

        def stream_generate(chat, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter(first_chunks)
            return iter(second_chunks)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=stream_generate)

        def tool_a():
            executed.append("a")
            return "result_a"

        def tool_b():
            executed.append("b")
            return "result_b"

        list(run_agentic_stream(client, "test", tools=[tool_a, tool_b]))

        assert executed == ["a", "b"]
        assert call_count[0] == 2

    def test_streaming_realistic_chunk_sequence_from_openai_format(self):
        """Streaming with realistic OpenAI-style chunk sequence including usage.

        This test uses chunk shapes that mirror actual OpenAI streaming responses,
        including the final chunk with usage information.
        """
        chunks = [
            ChatCompletionModel(
                id="chatcmpl-9XyZ1",
                model="gpt-4-0125-preview",
                choices=[ChoiceModel(index=0, delta=DeltaModel(content=""), finish_reason=None)],
                usage=StreamUsageModel(),
            ),
            ChatCompletionModel(
                id="chatcmpl-9XyZ2",
                model="gpt-4-0125-preview",
                choices=[ChoiceModel(index=0, delta=DeltaModel(content="The"), finish_reason=None)],
                usage=StreamUsageModel(),
            ),
            ChatCompletionModel(
                id="chatcmpl-9XyZ3",
                model="gpt-4-0125-preview",
                choices=[ChoiceModel(index=0, delta=DeltaModel(content=" answer"), finish_reason=None)],
                usage=StreamUsageModel(),
            ),
            ChatCompletionModel(
                id="chatcmpl-9XyZ4",
                model="gpt-4-0125-preview",
                choices=[ChoiceModel(index=0, delta=DeltaModel(content=" is"), finish_reason=None)],
                usage=StreamUsageModel(),
            ),
            ChatCompletionModel(
                id="chatcmpl-9XyZ5",
                model="gpt-4-0125-preview",
                choices=[ChoiceModel(index=0, delta=DeltaModel(content=" 42."), finish_reason="stop")],
                usage=StreamUsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        ]

        client = MagicMock()
        client.llm.stream_generate = MagicMock(return_value=iter(chunks))

        result_chunks = list(run_agentic_stream(client, "What is the answer?"))

        assert len(result_chunks) == 5
        # Verify chunk order preserved
        assert result_chunks[0].choices[0].delta.content == ""
        assert result_chunks[1].choices[0].delta.content == "The"
        assert result_chunks[2].choices[0].delta.content == " answer"
        assert result_chunks[3].choices[0].delta.content == " is"
        assert result_chunks[4].choices[0].delta.content == " 42."
        assert result_chunks[4].choices[0].finish_reason == "stop"
        # Final chunk has usage
        assert result_chunks[4].usage.total_tokens == 15


# ─── Adapter Expectation Integration Tests ────────────────────────────────

class TestAdapterExpectationIntegration:
    """Tests that show what provider adapters expect from the agentic loop.

    These document the contract between the agentic loop and provider adapters
    (OpenAIToolAdapter, AnthropicToolAdapter) using realistic data shapes.
    """

    def test_openai_adapter_expects_role_tool_messages(self):
        """OpenAIToolAdapter.serialize_tool_results expects ToolResult objects.

        This shows how the canonical ToolResult model maps to OpenAI's wire format.
        """
        results = [
            ToolResult(
                tool_call_id="call_abc123",
                name="get_weather",
                content='{"temperature": 18, "unit": "C"}',
                is_error=False,
                duration_ms=42.5,
            ),
        ]

        # Simulate OpenAIToolAdapter behavior:
        chat = ModelChat()
        for result in results:
            chat.add_tool_result(
                tool_call_id=result.tool_call_id,
                content=result.content,
                is_error=result.is_error,
            )

        assert len(chat.messages) == 1
        msg = chat.messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc123"
        assert msg["content"] == '{"temperature": 18, "unit": "C"}'
        assert msg["is_error"] is False

    def test_openai_adapter_expects_parallel_tool_results(self):
        """OpenAIToolAdapter produces N role:tool messages for N parallel results."""
        results = [
            ToolResult(tool_call_id="call_1", name="search", content="[result1]"),
            ToolResult(tool_call_id="call_2", name="filter", content="[result2]"),
            ToolResult(tool_call_id="call_3", name="rank", content="[result3]"),
        ]

        chat = ModelChat()
        for result in results:
            chat.add_tool_result(
                tool_call_id=result.tool_call_id,
                content=result.content,
                is_error=result.is_error,
            )

        assert len(chat.messages) == 3
        for i, msg in enumerate(chat.messages):
            assert msg["role"] == "tool"
            assert msg["tool_call_id"] == f"call_{i + 1}"

    def test_anthropic_adapter_expects_user_with_content_blocks(self):
        """AnthropicToolAdapter bundles tool results in a single user message with content blocks.

        This shows how the canonical ToolResult model maps to Anthropic's wire format.
        """
        results = [
            ToolResult(tool_call_id="toolu_01ABC", name="get_weather", content='{"temp": 18}'),
            ToolResult(tool_call_id="toolu_01DEF", name="translate", content='{"text": "..."}'),
        ]

        # Simulate AnthropicToolAdapter behavior:
        chat = ModelChat()
        chat.add_tool_messages([{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": r.tool_call_id,
                    "content": r.content,
                }
                for r in results
            ],
        }])

        assert len(chat.messages) == 1
        msg = chat.messages[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_01ABC"
        assert msg["content"][1]["tool_use_id"] == "toolu_01DEF"

    def test_canonical_tool_call_to_provider_tool_call_mapping(self):
        """CanonicalToolCall can be mapped to provider-specific ToolCall format.

        This documents the expected adapter contract for deserialization.
        """
        # Canonical form (what AgentLoop works with)
        canonical = CanonicalToolCall(
            id="call_abc123",
            name="get_weather",
            arguments={"city": "London", "unit": "celsius"},
        )

        # OpenAI provider form (what the provider returns)
        provider_tool_call = ToolCall(
            id=canonical.id,
            function=FunctionCall(
                name=canonical.name,
                arguments=json.dumps(canonical.arguments),
            ),
        )

        # Adapter should be able to convert between these
        assert provider_tool_call.id == canonical.id
        assert provider_tool_call.function.name == canonical.name
        # Provider arguments are JSON strings, canonical are dicts
        assert json.loads(provider_tool_call.function.arguments) == canonical.arguments

    def test_tool_result_error_format_for_adapters(self):
        """ToolResult error format is consistent across provider adapters.

        Both OpenAI and Anthropic adapters need error information in the
        tool result content.
        """
        error_result = ToolResult(
            tool_call_id="call_err",
            name="failing_tool",
            content='{"error": "connection timeout", "type": "ConnectionError"}',
            is_error=True,
            error="connection timeout",
            error_type="ConnectionError",
            duration_ms=30000.0,
        )

        # OpenAI adapter: role:tool with error content
        openai_chat = ModelChat()
        openai_chat.add_tool_result(
            tool_call_id=error_result.tool_call_id,
            content=error_result.content,
            is_error=error_result.is_error,
        )
        assert openai_chat.messages[0]["is_error"] is True

        # Anthropic adapter: user message with tool_result content block
        anthropic_chat = ModelChat()
        anthropic_chat.add_tool_messages([{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": error_result.tool_call_id,
                "content": error_result.content,
            }],
        }])
        assert anthropic_chat.messages[0]["content"][0]["tool_use_id"] == "call_err"


# ─── Integration: Realistic Multi-Turn Tool Workflow ──────────────────────

class TestRealisticMultiTurnWorkflow:
    """Integration tests simulating realistic multi-turn tool workflows.

    These mirror actual usage patterns like:
    - Search → Summarize → Answer
    - Query rewrite → Search → Rerank → Answer
    """

    def test_search_then_summarize_workflow(self):
        """Realistic workflow: search tool returns data, summarize tool processes it."""
        # First LLM call: decide to search
        search_call = _make_tool_call(
            name="search_papers",
            arguments='{"query":"machine learning education","limit":10}',
        )
        search_response = _make_response(content="I'll search for papers...", tool_calls=[search_call])

        # Second LLM call: decide to summarize
        summarize_call = _make_tool_call(
            name="summarize_results",
            arguments='{"papers":[{"title":"Paper 1"},{"title":"Paper 2"}]}',
        )
        summarize_response = _make_response(content="Now summarizing...", tool_calls=[summarize_call])

        # Third LLM call: final answer
        final_response = _make_response(
            content="Based on the search, here are the key findings:\n\n"
                    "1. Paper 1 discusses ML applications in education.\n"
                    "2. Paper 2 focuses on adaptive learning systems.",
            finish_reason="stop",
        )

        client = _make_mock_client([search_response, summarize_response, final_response])

        def search_papers(query, limit=10):
            return {"papers": [{"title": "Paper 1"}, {"title": "Paper 2"}], "count": 2}

        def summarize_results(papers):
            return f"Summarized {len(papers)} papers"

        iteration = [0]

        def capture_chat(chat, **kwargs):
            iteration[0] += 1
            if iteration[0] == 1:
                return search_response
            elif iteration[0] == 2:
                return summarize_response
            return final_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(
            client,
            "Find and summarize papers about ML in education",
            tools=[search_papers, summarize_results],
            max_iterations=5,
        )

        assert "Paper 1" in result.content
        assert "Paper 2" in result.content
        assert client.llm.generate.call_count == 3

    def test_query_rewrite_then_search_workflow(self):
        """Realistic workflow: rewrite ambiguous query, then search with variants."""
        # First: rewrite query
        rewrite_call = _make_tool_call(
            name="rewrite_query",
            arguments='{"text":"ai on education"}',
        )
        rewrite_response = _make_response(content="Let me refine your query...", tool_calls=[rewrite_call])

        # Second: search with variants
        search_call = _make_tool_call(
            name="search_papers",
            arguments='{"query":["artificial intelligence education", "AI in teaching", "ML learning systems"]}',
        )
        search_response = _make_response(content="Searching with variants...", tool_calls=[search_call])

        # Third: final answer
        final_response = _make_response(
            content="Found 15 relevant papers across your query variants.",
            finish_reason="stop",
        )

        client = _make_mock_client([rewrite_response, search_response, final_response])

        def rewrite_query(text):
            return {"variants": ["artificial intelligence education", "AI in teaching", "ML learning systems"]}

        def search_papers(query):
            queries = query if isinstance(query, list) else [query]
            return {"results": [{"title": f"Paper about {q}"} for q in queries], "total": 15}

        iteration = [0]

        def capture_chat(chat, **kwargs):
            iteration[0] += 1
            if iteration[0] == 1:
                return rewrite_response
            elif iteration[0] == 2:
                return search_response
            return final_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(
            client,
            "Find papers about ai on education",
            tools=[rewrite_query, search_papers],
            max_iterations=5,
        )

        assert "15" in result.content
        assert client.llm.generate.call_count == 3

    def test_workflow_with_tool_error_recovery(self):
        """Realistic workflow: tool fails, LLM sees error and retries or adapts."""
        # First: tool fails
        fail_call = _make_tool_call(name="fetch_data", arguments='{"url":"https://example.com"}')
        fail_response = _make_response(content="Fetching data...", tool_calls=[fail_call])

        # Second: LLM adapts and provides answer without tool
        final_response = _make_response(
            content="I couldn't fetch the data, but based on my knowledge...",
            finish_reason="stop",
        )

        client = _make_mock_client([fail_response, final_response])

        def fetch_data(url):
            raise ConnectionError("Network unreachable")

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                content = str(msg.get("content", ""))
                if "tool: fetch_data" in content and "Network unreachable" in content:
                    return final_response
            return fail_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(client, "Fetch data from example.com", tools=[fetch_data])

        assert "couldn't fetch" in result.content
        # Error feedback should be in chat history
        error_found = any(
            "Network unreachable" in str(m.get("content", ""))
            for m in captured_chat.messages
        )
        assert error_found
