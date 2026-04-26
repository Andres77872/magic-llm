"""Tests for the agentic loop (run_agentic) and streaming helpers.

TDD: agentic-tooling-flow-tests — Phase 1 + Phase 2 (Characterization)

Covers:
- Slice C: _accumulate_tool_calls_from_delta (Pure Unit)
- Slice D: _build_tool_calls_from_accumulated (Pure Unit)
- Slice E: run_agentic happy path — no tool calls (Characterization)
- Slice F: run_agentic — single tool call, one round (Characterization)
- Slice G: run_agentic — multi-tool in single iteration (Characterization)
- Slice H: run_agentic — unknown tool error handling (Characterization)
- Slice I: run_agentic — tool exception handling (Characterization)
- Slice J: run_agentic — max_iterations truncation (Characterization)
- Slice K: run_agentic — empty loop fallback (Characterization)
- Slice O: run_agentic — content concatenation (Characterization)

NOTE: All loop tests document CURRENT behavior. The SDD refactor will change
internal message semantics (role:"tool" vs role:"user"). These tests will need
updating when the refactor lands.
"""

import json
from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import (
    _accumulate_tool_calls_from_delta,
    _build_tool_calls_from_accumulated,
    run_agentic,
)
from magic_llm.model.ModelChatStream import ToolCall as StreamToolCall, FunctionCall as StreamFunctionCall


# ─── Slice C: _accumulate_tool_calls_from_delta ────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 1, Pure Unit

class TestAccumulateToolCallsFromDelta:
    """_accumulate_tool_calls_from_delta merges streaming tool call deltas by index."""

    def test_single_complete_call(self):
        accumulated = {}
        delta = [StreamToolCall(
            index=0,
            id="call_abc",
            function=StreamFunctionCall(name="get_weather", arguments='{"city":"London"}'),
        )]
        _accumulate_tool_calls_from_delta(accumulated, delta)
        assert 0 in accumulated
        assert accumulated[0]["id"] == "call_abc"
        assert accumulated[0]["function"]["name"] == "get_weather"
        assert accumulated[0]["function"]["arguments"] == '{"city":"London"}'

    def test_multi_call_by_index(self):
        accumulated = {}
        deltas = [
            StreamToolCall(index=0, id="call_1", function=StreamFunctionCall(name="tool_a", arguments='{}')),
            StreamToolCall(index=1, id="call_2", function=StreamFunctionCall(name="tool_b", arguments='{}')),
        ]
        _accumulate_tool_calls_from_delta(accumulated, deltas)
        assert len(accumulated) == 2
        assert accumulated[0]["function"]["name"] == "tool_a"
        assert accumulated[1]["function"]["name"] == "tool_b"

    def test_fragmented_argument_concatenation(self):
        """Arguments arrive in fragments across multiple deltas."""
        accumulated = {}
        # First fragment
        _accumulate_tool_calls_from_delta(accumulated, [
            StreamToolCall(index=0, id="call_1", function=StreamFunctionCall(name="get_weather", arguments='{"city')),
        ])
        # Second fragment
        _accumulate_tool_calls_from_delta(accumulated, [
            StreamToolCall(index=0, function=StreamFunctionCall(arguments='":"London"}')),
        ])
        assert accumulated[0]["function"]["arguments"] == '{"city":"London"}'

    def test_empty_delta_returns_early(self):
        accumulated = {"existing": {"id": "x"}}
        _accumulate_tool_calls_from_delta(accumulated, None)
        _accumulate_tool_calls_from_delta(accumulated, [])
        assert len(accumulated) == 1  # unchanged

    def test_id_propagation(self):
        """ID may arrive in a later delta than the name."""
        accumulated = {}
        _accumulate_tool_calls_from_delta(accumulated, [
            StreamToolCall(index=0, function=StreamFunctionCall(name="fn")),
        ])
        _accumulate_tool_calls_from_delta(accumulated, [
            StreamToolCall(index=0, id="call_late"),
        ])
        assert accumulated[0]["id"] == "call_late"
        assert accumulated[0]["function"]["name"] == "fn"

    def test_default_index_zero(self):
        """If index is None, defaults to 0."""
        accumulated = {}
        _accumulate_tool_calls_from_delta(accumulated, [
            StreamToolCall(index=None, id="call_x", function=StreamFunctionCall(name="fn", arguments='{}')),
        ])
        assert 0 in accumulated


# ─── Slice D: _build_tool_calls_from_accumulated ───────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 1, Pure Unit

class TestBuildToolCallsFromAccumulated:
    """_build_tool_calls_from_accumulated converts accumulated dict to ToolCall objects."""

    def test_single_entry(self):
        from magic_llm.model.ModelChatResponse import ToolCall
        accumulated = {
            0: {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"London"}'}},
        }
        result = _build_tool_calls_from_accumulated(accumulated)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].id == "call_1"
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"city":"London"}'

    def test_multi_entry_sorted_by_index(self):
        accumulated = {
            2: {"id": "call_3", "type": "function", "function": {"name": "tool_c", "arguments": '{}'}},
            0: {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": '{}'}},
            1: {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": '{}'}},
        }
        result = _build_tool_calls_from_accumulated(accumulated)
        assert [tc.function.name for tc in result] == ["tool_a", "tool_b", "tool_c"]

    def test_empty_dict(self):
        result = _build_tool_calls_from_accumulated({})
        assert result == []


# ─── Slice E: run_agentic happy path — no tool calls ──────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticNoToolCalls:
    """run_agentic returns content from a single LLM call when no tool_calls present."""

    def test_returns_content_from_single_call(self):
        response = _make_response(content="The weather is sunny.", finish_reason="stop")
        client = _make_mock_client([response])

        result = run_agentic(client, "What's the weather?")

        assert result.content == "The weather is sunny."
        assert client.llm.generate.call_count == 1

    def test_chat_has_user_and_assistant_messages(self):
        from magic_llm.model.ModelChat import ModelChat
        response = _make_response(content="Hello!", finish_reason="stop")
        client = _make_mock_client([response])

        # Capture the chat object passed to generate
        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            return response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "Hi there")

        assert captured_chat is not None
        roles = [m["role"] for m in captured_chat.messages]
        assert "user" in roles
        assert "assistant" in roles


# ─── Slice F: run_agentic — single tool call, one round ───────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticSingleToolCall:
    """run_agentic executes one tool call and continues to a final response."""

    def test_tool_executed_with_correct_args(self):
        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="It's 18C in London.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        called_with = {}
        def get_weather(city):
            called_with["city"] = city
            return {"temp": 18, "unit": "C"}

        result = run_agentic(client, "Weather in London?", tools=[get_weather])

        assert called_with.get("city") == "London"
        assert result.content == "It's 18C in London."

    def test_feedback_format_matches_format_tool_feedback(self):
        """Tool feedback is injected as role:user with _format_tool_feedback text."""
        from magic_llm.util.agentic import _format_tool_feedback
        from magic_llm.model.ModelChat import ModelChat

        tool_call = _make_tool_call(name="echo", arguments='{"msg":"hello"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def echo(msg):
            return f"Echo: {msg}"

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            # Return final_response on second call
            if captured_chat.messages[-1]["role"] == "user" and "tool:" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "Echo hello", tools=[echo])

        # Find the tool feedback message
        tool_msg = None
        for msg in captured_chat.messages:
            if msg["role"] == "user" and "tool: echo" in str(msg.get("content", "")):
                tool_msg = msg
                break

        assert tool_msg is not None
        expected = _format_tool_feedback(
            name="echo",
            input_str='{"msg": "hello"}',
            output_str="Echo: hello",
        )
        assert expected in tool_msg["content"]

    def test_loop_makes_two_generate_calls(self):
        tool_call = _make_tool_call(name="fn", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Final", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def fn():
            return "ok"

        run_agentic(client, "test", tools=[fn])

        assert client.llm.generate.call_count == 2


# ─── Slice G: run_agentic — multi-tool in single iteration ────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticMultiTool:
    """run_agentic executes multiple tool calls sequentially in one iteration."""

    def test_both_tools_executed(self):
        call_a = _make_tool_call(id="call_1", name="tool_a", arguments='{}')
        call_b = _make_tool_call(id="call_2", name="tool_b", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[call_a, call_b])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        executed = []
        def tool_a():
            executed.append("a")
            return "result_a"

        def tool_b():
            executed.append("b")
            return "result_b"

        run_agentic(client, "test", tools=[tool_a, tool_b])

        assert executed == ["a", "b"]

    def test_results_combined_with_separator(self):
        """Multi-tool results are joined with \\n\\n into a single user message."""
        call_a = _make_tool_call(id="call_1", name="tool_a", arguments='{}')
        call_b = _make_tool_call(id="call_2", name="tool_b", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[call_a, call_b])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def tool_a():
            return "result_a"

        def tool_b():
            return "result_b"

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if captured_chat.messages[-1]["role"] == "user" and "tool: tool_a" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "test", tools=[tool_a, tool_b])

        # Find the combined feedback message
        combined_msg = None
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            if "tool: tool_a" in content and "tool: tool_b" in content:
                combined_msg = msg
                break

        assert combined_msg is not None
        assert "\n\n" in combined_msg["content"]


# ─── Slice H: run_agentic — unknown tool error handling ───────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticUnknownTool:
    """run_agentic handles unknown tools by injecting error feedback and continuing."""

    def test_error_feedback_contains_unknown_tool_message(self):
        tool_call = _make_tool_call(name="nonexistent_tool", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="I cannot help with that.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if captured_chat.messages[-1]["role"] == "user" and "Unknown tool" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "test", tools=[])

        error_msg = None
        for msg in captured_chat.messages:
            if "Unknown tool: nonexistent_tool" in str(msg.get("content", "")):
                error_msg = msg
                break

        assert error_msg is not None

    def test_loop_continues_after_unknown_tool(self):
        tool_call = _make_tool_call(name="missing", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Final", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        run_agentic(client, "test", tools=[])

        # Should have made 2 generate calls: one for tool, one for final
        assert client.llm.generate.call_count == 2


# ─── Slice I: run_agentic — tool exception handling ───────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticToolException:
    """run_agentic catches tool exceptions and continues with error feedback."""

    def test_exception_caught_and_error_feedback_injected(self):
        tool_call = _make_tool_call(name="boom_tool", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Recovered", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def boom_tool():
            raise ValueError("boom")

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if captured_chat.messages[-1]["role"] == "user" and "boom" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        result = run_agentic(client, "test", tools=[boom_tool])

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
        tool_call = _make_tool_call(name="fail", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Final", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def fail():
            raise RuntimeError("fail")

        run_agentic(client, "test", tools=[fail])

        assert client.llm.generate.call_count == 2


# ─── Slice J: run_agentic — max_iterations truncation ─────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticMaxIterations:
    """run_agentic silently exits when max_iterations limit is hit."""

    def test_loop_makes_exactly_n_generate_calls(self):
        """When tool_calls are always present, loop stops at max_iterations."""
        tool_call = _make_tool_call(name="loop_tool", arguments='{}')
        tool_response = _make_response(content="thinking...", tool_calls=[tool_call])

        client = _make_mock_client([])
        # Always return tool_response
        client.llm.generate = MagicMock(return_value=tool_response)

        def loop_tool():
            return "ok"

        # max_iterations=3: should make exactly 3 generate calls
        result = run_agentic(client, "test", tools=[loop_tool], max_iterations=3)

        assert client.llm.generate.call_count == 3

    def test_no_exception_raised_on_truncation(self):
        tool_call = _make_tool_call(name="loop_tool", arguments='{}')
        tool_response = _make_response(content="thinking...", tool_calls=[tool_call])

        client = MagicMock()
        client.llm.generate = MagicMock(return_value=tool_response)

        def loop_tool():
            return "ok"

        # Should NOT raise — silent exit
        result = run_agentic(client, "test", tools=[loop_tool], max_iterations=2)
        assert result is not None

    def test_collected_content_from_all_iterations_present(self):
        tool_call = _make_tool_call(name="loop_tool", arguments='{}')
        responses = [
            _make_response(content="step 1", tool_calls=[tool_call]),
            _make_response(content="step 2", tool_calls=[tool_call]),
            _make_response(content="final", finish_reason="stop"),
        ]
        client = _make_mock_client(responses)

        def loop_tool():
            return "ok"

        result = run_agentic(client, "test", tools=[loop_tool], max_iterations=5)

        # Content should contain both "step 1" and "step 2" joined by separator
        assert "step 1" in result.content
        assert "step 2" in result.content


# ─── Slice K: run_agentic — empty loop fallback ───────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticEmptyLoopFallback:
    """run_agentic calls a final generate() when max_iterations=0 produces no responses."""

    def test_fallback_generate_called_when_max_iterations_zero(self):
        fallback_response = _make_response(content="Fallback response", finish_reason="stop")
        client = MagicMock()
        client.llm.generate = MagicMock(return_value=fallback_response)

        result = run_agentic(client, "test", max_iterations=0)

        # With max_iterations=0, the loop body never runs, last_response is None,
        # so the fallback generate is called
        assert client.llm.generate.call_count == 1
        assert result.content == "Fallback response"

    def test_fallback_response_returned(self):
        fallback_response = _make_response(content="Fallback", finish_reason="stop")
        client = MagicMock()
        client.llm.generate = MagicMock(return_value=fallback_response)

        result = run_agentic(client, "test", max_iterations=0)

        assert result.content == "Fallback"


# ─── Slice O: run_agentic — content concatenation ─────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2, Characterization

class TestRunAgenticContentConcatenation:
    """run_agentic joins content from multiple iterations with content_separator."""

    def test_content_joined_with_separator(self):
        tool_call = _make_tool_call(name="fn", arguments='{}')
        responses = [
            _make_response(content="first part", tool_calls=[tool_call]),
            _make_response(content="second part", tool_calls=[tool_call]),
            _make_response(content="third part", finish_reason="stop"),
        ]
        client = _make_mock_client(responses)

        def fn():
            return "ok"

        result = run_agentic(client, "test", tools=[fn], max_iterations=5)

        assert "first part" in result.content
        assert "second part" in result.content
        assert "third part" in result.content

    def test_custom_separator_used(self):
        tool_call = _make_tool_call(name="fn", arguments='{}')
        responses = [
            _make_response(content="A", tool_calls=[tool_call]),
            _make_response(content="B", finish_reason="stop"),
        ]
        client = _make_mock_client(responses)

        def fn():
            return "ok"

        result = run_agentic(client, "test", tools=[fn], content_separator=" | ")

        assert "A | B" == result.content


# ─── Internal helpers ──────────────────────────────────────────────────────

def _make_mock_client(responses: list):
    """Create a mock client that returns responses in sequence."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.llm.generate = MagicMock(side_effect=responses)
    return client


def _make_response(content=None, tool_calls=None, finish_reason="stop", model="test-model"):
    """Build a valid ModelChatResponse."""
    from magic_llm.model.ModelChatResponse import (
        ModelChatResponse, Choice, Message, UsageModel,
    )
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="test-1",
        object="chat.completion",
        created=1700000000.0,
        model=model,
        choices=[choice],
        usage=UsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_tool_call(id="call_1", name="get_weather", arguments='{"city":"London"}'):
    """Build a valid ToolCall."""
    from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))
