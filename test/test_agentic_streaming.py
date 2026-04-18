"""Tests for the streaming agentic loop (run_agentic_stream).

TDD: agentic-tooling-flow-tests — Phase 4 (Streaming Unit)

Covers:
- Slice T2: Streaming happy path — no tool calls
- Slice T:  Streaming delta accumulation and tool-call completeness
- Slice T3: Streaming single tool call full cycle
- Slice U:  Streaming separator between iterations
- Streaming unknown tool handling
- Streaming tool exception handling
- Streaming multi-tool accumulation and execution
- Streaming tool result reinsertion behavior
- Helper coverage: _create_separator_chunk edge cases

NOTE: All tests document CURRENT behavior against run_agentic_stream().
The SDD refactor will change internal message semantics. These tests will
need updating when the refactor lands.
"""

import json
from typing import Iterator, List
from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import (
    _accumulate_tool_calls_from_delta,
    _build_tool_calls_from_accumulated,
    _create_separator_chunk,
    run_agentic_stream,
)
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    ChoiceModel,
    DeltaModel,
    ToolCall as StreamToolCall,
    FunctionCall as StreamFunctionCall,
)


# ─── Streaming Chunk Builders ──────────────────────────────────────────────

def _make_chunk(
    content: str = None,
    tool_calls: List[StreamToolCall] = None,
    finish_reason: str = None,
    chunk_id: str = "chunk-1",
    model: str = "test-model",
) -> ChatCompletionModel:
    """Build a valid ChatCompletionModel stream chunk."""
    delta = DeltaModel(content=content, tool_calls=tool_calls)
    choice = ChoiceModel(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionModel(
        id=chunk_id,
        model=model,
        choices=[choice],
    )


def _make_tool_delta(
    index: int = 0,
    id: str = None,
    name: str = None,
    arguments: str = None,
) -> StreamToolCall:
    """Build a streaming ToolCall delta (for delta.tool_calls)."""
    return StreamToolCall(
        index=index,
        id=id,
        function=StreamFunctionCall(name=name, arguments=arguments),
    )


def _make_stream_client(chunks: List[ChatCompletionModel]) -> MagicMock:
    """Create a mock client whose stream_generate yields the given chunks."""
    client = MagicMock()
    client.llm.stream_generate = MagicMock(return_value=iter(chunks))
    return client


def _make_multi_iter_stream_client(
    iter_chunks: List[List[ChatCompletionModel]],
) -> MagicMock:
    """Create a mock client that yields different chunk sequences per iteration.

    Each call to stream_generate returns the next list of chunks.
    """
    client = MagicMock()
    iterators = [iter(chunks) for chunks in iter_chunks]
    client.llm.stream_generate = MagicMock(side_effect=iterators)
    return client


# ─── Slice T2: Streaming happy path — no tool calls ───────────────────────

class TestRunAgenticStreamNoToolCalls:
    """run_agentic_stream yields all chunks and completes after one iteration when no tool_calls."""

    def test_yields_all_chunks_in_order(self):
        chunks = [
            _make_chunk(content="Hello", chunk_id="c1"),
            _make_chunk(content=" ", chunk_id="c2"),
            _make_chunk(content="world", chunk_id="c3"),
            _make_chunk(content="!", chunk_id="c4", finish_reason="stop"),
        ]
        client = _make_stream_client(chunks)

        result = list(run_agentic_stream(client, "Say hello"))

        assert len(result) == 4
        assert result[0].choices[0].delta.content == "Hello"
        assert result[1].choices[0].delta.content == " "
        assert result[2].choices[0].delta.content == "world"
        assert result[3].choices[0].delta.content == "!"

    def test_no_tool_execution(self):
        """No tools should be called when the stream has no tool_calls."""
        chunks = [
            _make_chunk(content="Done", finish_reason="stop"),
        ]
        client = _make_stream_client(chunks)

        called = []
        def my_tool():
            called.append(True)
            return "ok"

        list(run_agentic_stream(client, "test", tools=[my_tool]))

        assert called == []

    def test_stream_generate_called_once(self):
        chunks = [_make_chunk(content="ok", finish_reason="stop")]
        client = _make_stream_client(chunks)

        list(run_agentic_stream(client, "test"))

        assert client.llm.stream_generate.call_count == 1

    def test_empty_stream_completes_without_error(self):
        """An empty stream (no chunks) should complete without error."""
        client = _make_stream_client([])

        result = list(run_agentic_stream(client, "test"))

        assert result == []


# ─── Slice T: Streaming delta accumulation completeness ───────────────────

class TestRunAgenticStreamDeltaAccumulation:
    """Tool calls are only executed after full stream completion, with fragmented args reassembled."""

    def test_fragmented_arguments_correctly_reassembled(self):
        """Arguments arrive in fragments across multiple deltas; tool receives full JSON."""
        # Stream: content + fragmented tool call across 3 chunks
        chunks = [
            _make_chunk(content="Let me check", chunk_id="c1"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="get_weather", arguments='{"city')],
                chunk_id="c2",
            ),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, arguments='":"London"}')],
                chunk_id="c3",
                finish_reason="tool_calls",
            ),
        ]
        # After stream ends, tool executes, then a second iteration with final response
        final_chunks = [
            _make_chunk(content="It's 18C.", chunk_id="c4", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([chunks, final_chunks])

        received_args = {}
        def get_weather(city):
            received_args["city"] = city
            return {"temp": 18}

        result = list(run_agentic_stream(client, "Weather?", tools=[get_weather]))

        assert received_args.get("city") == "London"

    def test_tool_executes_after_stream_not_during(self):
        """Tool should NOT be called while chunks are still being yielded."""
        call_order = []

        def get_weather(city="London"):
            call_order.append("tool")
            return {"temp": 18}

        # First iteration: content + tool call (content is truthy → separator emitted)
        chunks = [
            _make_chunk(content="Checking", chunk_id="c1", model="test-model"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="get_weather", arguments='{}')],
                chunk_id="c2",
                model="test-model",
            ),
        ]
        # Second iteration: final response
        final_chunks = [
            _make_chunk(content="Done", chunk_id="c3", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([chunks, final_chunks])

        result = list(run_agentic_stream(client, "test", tools=[get_weather]))

        # c1, c2, separator, c3 — tool executes between iteration 1 and 2
        assert len(result) == 4
        assert call_order == ["tool"]
        # Separator is at index 2
        assert result[2].choices[0].delta.content == "\n\n"

    def test_multi_tool_accumulation_by_index(self):
        """Multiple tool calls accumulated by their index across deltas."""
        chunks = [
            _make_chunk(
                content="",
                tool_calls=[
                    _make_tool_delta(index=0, id="call_a", name="tool_a", arguments='{}'),
                    _make_tool_delta(index=1, id="call_b", name="tool_b", arguments='{}'),
                ],
                chunk_id="c1",
            ),
        ]
        final_chunks = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([chunks, final_chunks])

        executed = []
        def tool_a():
            executed.append("a")
            return "result_a"
        def tool_b():
            executed.append("b")
            return "result_b"

        list(run_agentic_stream(client, "test", tools=[tool_a, tool_b]))

        assert executed == ["a", "b"]


# ─── Slice T3: Streaming single tool call full cycle ──────────────────────

class TestRunAgenticStreamSingleToolCall:
    """run_agentic_stream full cycle: chunks → tool executes → separator → second iteration chunks."""

    def test_chunks_yielded_tool_executes_separator_yielded_second_iteration(self):
        """Full cycle: first iteration chunks, tool execution, separator, second iteration chunks."""
        # First iteration: content + tool call
        first_iter = [
            _make_chunk(content="I'll check", chunk_id="c1", model="test-model"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="get_weather", arguments='{"city":"London"}')],
                chunk_id="c2",
                model="test-model",
            ),
        ]
        # Second iteration: final response
        second_iter = [
            _make_chunk(content="It's 18C.", chunk_id="c3", model="test-model", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def get_weather(city):
            return {"temp": 18}

        result = list(run_agentic_stream(client, "Weather in London?", tools=[get_weather]))

        # Should have: c1, c2, separator, c3
        assert len(result) == 4
        assert result[0].id == "c1"
        assert result[1].id == "c2"
        # result[2] is the separator chunk
        assert result[3].id == "c3"
        assert result[3].choices[0].delta.content == "It's 18C."

    def test_tool_result_reinserted_as_user_message(self):
        """After tool execution, result is reinserted as a single user message (current behavior)."""
        from magic_llm.model.ModelChat import ModelChat
        from magic_llm.util.agentic import _format_tool_feedback

        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="echo", arguments='{"msg":"hello"}')],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and chat.messages[-1]["role"] == "user":
                return iter(second_iter)
            return iter(first_iter)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        def echo(msg):
            return f"Echo: {msg}"

        list(run_agentic_stream(client, "Echo hello", tools=[echo]))

        # Find the tool feedback message
        tool_msg = None
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            if "tool: echo" in content:
                tool_msg = msg
                break

        assert tool_msg is not None
        assert tool_msg["role"] == "user"
        expected = _format_tool_feedback(
            name="echo",
            input_str='{"msg": "hello"}',
            output_str="Echo: hello",
        )
        assert expected in tool_msg["content"]


# ─── Slice U: Streaming separator between iterations ──────────────────────

class TestRunAgenticStreamSeparator:
    """Separator chunk is yielded between tool-execution iterations."""

    def test_separator_chunk_yielded_between_iterations(self):
        """When tool calls are present, a separator chunk is yielded between iterations."""
        first_iter = [
            _make_chunk(content="Thinking", chunk_id="c1", model="test-model"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                chunk_id="c2",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c3", model="test-model", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def fn():
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn]))

        # c1, c2, separator, c3
        assert len(result) == 4
        separator = result[2]
        assert separator.choices[0].delta.content == "\n\n"

    def test_separator_content_matches_content_separator_param(self):
        """Custom content_separator is reflected in the separator chunk."""
        first_iter = [
            _make_chunk(content="A", chunk_id="c1", model="test-model"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                chunk_id="c2",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="B", chunk_id="c3", model="test-model", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def fn():
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn], content_separator=" --- "))

        assert len(result) == 4
        separator = result[2]
        assert separator.choices[0].delta.content == " --- "

    def test_no_separator_when_no_tool_calls(self):
        """No separator chunk should be yielded when the stream has no tool_calls."""
        chunks = [
            _make_chunk(content="Hello", chunk_id="c1"),
            _make_chunk(content=" world", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_stream_client(chunks)

        result = list(run_agentic_stream(client, "test"))

        assert len(result) == 2
        # No separator chunk should be present
        for chunk in result:
            assert chunk.choices[0].delta.content != "\n\n"

    def test_no_separator_when_iteration_content_empty(self):
        """Separator is only yielded when iteration_content is truthy (line 410 check)."""
        # First iteration: ONLY tool call delta, no content
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                chunk_id="c1",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", model="test-model", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def fn():
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn]))

        # Should be: c1, c2 (no separator because iteration_content is empty)
        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c2"


# ─── Streaming: unknown tool handling ─────────────────────────────────────

class TestRunAgenticStreamUnknownTool:
    """run_agentic_stream handles unknown tools by injecting error feedback and continuing."""

    def test_error_feedback_contains_unknown_tool_message(self):
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="nonexistent_tool", arguments='{}')],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="I cannot help.", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and "Unknown tool" in str(chat.messages[-1]):
                return iter(second_iter)
            return iter(first_iter)

        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        result = list(run_agentic_stream(client, "test", tools=[]))

        # Find error feedback
        error_msg = None
        for msg in captured_chat.messages:
            if "Unknown tool: nonexistent_tool" in str(msg.get("content", "")):
                error_msg = msg
                break

        assert error_msg is not None

    def test_loop_continues_after_unknown_tool(self):
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="missing", arguments='{}')],
                chunk_id="c1",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Final", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        result = list(run_agentic_stream(client, "test", tools=[]))

        # No separator because iteration_content is empty (only tool_call delta, no content)
        # Line 410: if iteration_content and content_separator:
        assert len(result) == 2  # c1, c2


# ─── Streaming: tool exception handling ───────────────────────────────────

class TestRunAgenticStreamToolException:
    """run_agentic_stream catches tool exceptions and continues with error feedback."""

    def test_exception_caught_and_error_feedback_injected(self):
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="boom_tool", arguments='{}')],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="Recovered", chunk_id="c2", finish_reason="stop"),
        ]

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and "boom" in str(chat.messages[-1]):
                return iter(second_iter)
            return iter(first_iter)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        def boom_tool():
            raise ValueError("boom")

        result = list(run_agentic_stream(client, "test", tools=[boom_tool]))

        # Find error feedback
        error_msg = None
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            if "tool: boom_tool" in content and "boom" in content:
                error_msg = msg
                break

        assert error_msg is not None
        assert "boom" in error_msg["content"]

    def test_loop_continues_after_tool_exception(self):
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fail", arguments='{}')],
                chunk_id="c1",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Final", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def fail():
            raise RuntimeError("fail")

        result = list(run_agentic_stream(client, "test", tools=[fail]))

        # No separator because iteration_content is empty
        assert len(result) == 2  # c1, c2


# ─── Streaming: multi-tool accumulation and execution ─────────────────────

class TestRunAgenticStreamMultiTool:
    """run_agentic_stream handles multiple tool calls in a single streaming iteration."""

    def test_both_tools_executed_sequentially(self):
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[
                    _make_tool_delta(index=0, id="call_a", name="tool_a", arguments='{}'),
                    _make_tool_delta(index=1, id="call_b", name="tool_b", arguments='{}'),
                ],
                chunk_id="c1",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        executed = []
        def tool_a():
            executed.append("a")
            return "result_a"
        def tool_b():
            executed.append("b")
            return "result_b"

        list(run_agentic_stream(client, "test", tools=[tool_a, tool_b]))

        assert executed == ["a", "b"]

    def test_results_combined_with_double_newline(self):
        """Multi-tool results are joined with \\n\\n into a single user message."""
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[
                    _make_tool_delta(index=0, id="call_a", name="tool_a", arguments='{}'),
                    _make_tool_delta(index=1, id="call_b", name="tool_b", arguments='{}'),
                ],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and "tool: tool_a" in str(chat.messages[-1]):
                return iter(second_iter)
            return iter(first_iter)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        def tool_a():
            return "result_a"
        def tool_b():
            return "result_b"

        list(run_agentic_stream(client, "test", tools=[tool_a, tool_b]))

        # Find the combined feedback message
        combined_msg = None
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            if "tool: tool_a" in content and "tool: tool_b" in content:
                combined_msg = msg
                break

        assert combined_msg is not None
        assert "\n\n" in combined_msg["content"]


# ─── Streaming: tool result reinsertion behavior ──────────────────────────

class TestRunAgenticStreamToolResultReinsertion:
    """Documents current behavior: tool results reinserted as role:user plain text."""

    def test_tool_result_is_role_user_not_role_tool(self):
        """After tool execution, chat.messages contains role='user' (NOT role='tool')."""
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="get_weather", arguments='{"city":"London"}')],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and chat.messages[-1]["role"] == "user":
                return iter(second_iter)
            return iter(first_iter)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        def get_weather(city):
            return {"temp": 18}

        list(run_agentic_stream(client, "Weather?", tools=[get_weather]))

        # Find tool result message
        tool_msg = None
        for msg in captured_chat.messages:
            if "tool: get_weather" in str(msg.get("content", "")):
                tool_msg = msg
                break

        assert tool_msg is not None
        assert tool_msg["role"] == "user"
        # Verify it's NOT role:tool (current behavior)
        assert not any(
            m.get("role") == "tool" for m in captured_chat.messages
        )

    def test_single_user_message_for_all_tool_results(self):
        """All tool results from one iteration are combined into ONE user message."""
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[
                    _make_tool_delta(index=0, id="call_a", name="tool_a", arguments='{}'),
                    _make_tool_delta(index=1, id="call_b", name="tool_b", arguments='{}'),
                ],
                chunk_id="c1",
            ),
        ]
        second_iter = [
            _make_chunk(content="Done", chunk_id="c2", finish_reason="stop"),
        ]

        captured_chat = None
        def capture_stream(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if len(chat.messages) > 1 and "tool: tool_a" in str(chat.messages[-1]):
                return iter(second_iter)
            return iter(first_iter)

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=capture_stream)

        def tool_a():
            return "a"
        def tool_b():
            return "b"

        list(run_agentic_stream(client, "test", tools=[tool_a, tool_b]))

        # Count user messages that contain tool feedback
        tool_user_messages = [
            msg for msg in captured_chat.messages
            if msg.get("role") == "user" and "tool:" in str(msg.get("content", ""))
        ]
        assert len(tool_user_messages) == 1


# ─── Helper: _create_separator_chunk edge cases ───────────────────────────

class TestCreateSeparatorChunkEdgeCases:
    """_create_separator_chunk handles edge cases for model/id fallback."""

    def test_normal_construction(self):
        chunk = _create_separator_chunk(separator="\n\n", model="gpt-4", chunk_id="sep-1")

        assert isinstance(chunk, ChatCompletionModel)
        assert chunk.id == "sep-1"
        assert chunk.model == "gpt-4"
        assert chunk.choices[0].delta.content == "\n\n"

    def test_empty_model_fallback(self):
        chunk = _create_separator_chunk(separator=" --- ", model="", chunk_id="sep-2")

        assert chunk.model == ""
        assert chunk.choices[0].delta.content == " --- "

    def test_empty_separator(self):
        chunk = _create_separator_chunk(separator="", model="test-model", chunk_id="sep-3")

        assert chunk.choices[0].delta.content == ""

    def test_chunk_is_valid_chat_completion_model(self):
        chunk = _create_separator_chunk(separator="\n\n", model="test", chunk_id="sep-4")

        assert hasattr(chunk, "id")
        assert hasattr(chunk, "model")
        assert hasattr(chunk, "choices")
        assert len(chunk.choices) == 1
        assert hasattr(chunk.choices[0], "delta")


# ─── Streaming: max_iterations behavior ───────────────────────────────────

class TestRunAgenticStreamMaxIterations:
    """run_agentic_stream respects max_iterations cap."""

    def test_loop_stops_at_max_iterations(self):
        """When tool_calls are always present, loop stops at max_iterations."""
        # Each call to stream_generate returns a fresh iterator with a tool call
        def make_tool_iter(*args, **kwargs):
            return iter([
                _make_chunk(
                    content="",
                    tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                    chunk_id="c1",
                    model="test-model",
                ),
            ])

        client = MagicMock()
        client.llm.stream_generate = MagicMock(side_effect=make_tool_iter)

        def fn():
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn], max_iterations=2))

        # Iter 1: c1 (no separator — iteration_content is empty)
        # Iter 2: c1 (loop exits because range(2) is exhausted)
        # No separator in either iteration because iteration_content is empty
        assert len(result) == 2
        assert client.llm.stream_generate.call_count == 2

    def test_no_exception_raised_on_truncation(self):
        """max_iterations truncation should NOT raise an exception."""
        tool_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                chunk_id="c1",
            ),
        ]
        client = MagicMock()
        client.llm.stream_generate = MagicMock(return_value=iter(tool_iter))

        def fn():
            return "ok"

        # Should NOT raise
        result = list(run_agentic_stream(client, "test", tools=[fn], max_iterations=1))

        assert result is not None


# ─── Streaming: content accumulation across iterations ────────────────────

class TestRunAgenticStreamContentAccumulation:
    """run_agentic_stream accumulates content across streaming iterations."""

    def test_content_from_multiple_iterations_accumulated(self):
        """Content from each iteration is collected and accumulated in final response."""
        first_iter = [
            _make_chunk(content="First", chunk_id="c1", model="test-model"),
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="fn", arguments='{}')],
                chunk_id="c2",
                model="test-model",
            ),
        ]
        second_iter = [
            _make_chunk(content="Second", chunk_id="c3", model="test-model", finish_reason="stop"),
        ]
        client = _make_multi_iter_stream_client([first_iter, second_iter])

        def fn():
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn]))

        # Verify all chunks are present
        contents = [chunk.choices[0].delta.content for chunk in result]
        assert "First" in contents
        assert "Second" in contents


# ─── Streaming: malformed tool_call handling ──────────────────────────────

class TestRunAgenticStreamMalformedToolCall:
    """run_agentic_stream skips malformed tool calls (no name, no function)."""

    def test_skips_tool_call_with_no_name(self):
        """Tool call with empty name should be skipped."""
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[_make_tool_delta(index=0, id="call_1", name="", arguments='{}')],
                chunk_id="c1",
            ),
        ]
        # Since no valid tool calls, loop should end after first iteration
        client = _make_stream_client(first_iter)

        called = []
        def fn():
            called.append(True)
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn]))

        # No tool should be called
        assert called == []
        # Only the first iteration chunks
        assert len(result) == 1

    def test_skips_tool_call_with_no_function(self):
        """Tool call with None function should be skipped."""
        first_iter = [
            _make_chunk(
                content="",
                tool_calls=[StreamToolCall(index=0, id="call_1", function=None)],
                chunk_id="c1",
            ),
        ]
        client = _make_stream_client(first_iter)

        called = []
        def fn():
            called.append(True)
            return "ok"

        result = list(run_agentic_stream(client, "test", tools=[fn]))

        assert called == []
        assert len(result) == 1
