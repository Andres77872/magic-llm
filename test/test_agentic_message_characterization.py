"""Characterization tests for agentic loop message semantics.

TDD: agentic-tooling-flow-tests — Phase 2b (Message Structure Characterization)

⚠️  DOCUMENTS CURRENT BEHAVIOR — WILL CHANGE WHEN SDD REFACTOR LANDS ⚠️

These tests explicitly document the NON-STRICT message format used by the current
run_agentic() implementation:

1. Tool results are injected as role:"user" plain text (NOT role:"tool").
2. Assistant tool_calls array is DISCARDED from chat history.
3. Multi-tool results are COMBINED into a single user message.

When the SDD refactor (react-like-agent-loop-refactor) replaces the loop with
AgentLoop using proper tool semantics, these tests WILL FAIL. That is expected
and intentional — they serve as a migration checklist.

Covers:
- Slice L: Tool results as role:user (Characterization)
- Slice M: Assistant tool_calls discarded (Characterization)
- Slice N: Combined feedback format (Characterization)
"""

from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import run_agentic, _format_tool_feedback


# ─── Slice L: Tool results as role:user ────────────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2b, Characterization

class TestToolResultsAsUserText:
    """CURRENT BEHAVIOR: Tool results are injected as role:"user" plain text.

    After SDD refactor, these should become role:"tool" with tool_call_id.
    """

    def test_tool_feedback_is_role_user_not_role_tool(self):
        """After tool execution, chat.messages contains role='user' NOT role='tool'."""
        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="It's 18C.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18}

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            # On second call (after tool feedback), return final
            if captured_chat.messages[-1]["role"] == "user" and "tool: get_weather" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "Weather?", tools=[get_weather])

        # Find the tool feedback message
        tool_feedback_msg = None
        for msg in captured_chat.messages:
            if msg["role"] == "user" and "tool: get_weather" in str(msg.get("content", "")):
                tool_feedback_msg = msg
                break

        assert tool_feedback_msg is not None, "Tool feedback message not found in chat"
        # CRITICAL: current behavior is role="user", NOT role="tool"
        assert tool_feedback_msg["role"] == "user"

        # Verify NO role:"tool" messages exist
        tool_role_messages = [m for m in captured_chat.messages if m["role"] == "tool"]
        assert len(tool_role_messages) == 0, (
            f"Expected NO role:'tool' messages (current behavior), "
            f"but found {len(tool_role_messages)}"
        )

    def test_tool_feedback_content_matches_format_tool_feedback(self):
        """Content of the user message matches _format_tool_feedback output."""
        tool_call = _make_tool_call(name="echo", arguments='{"msg":"hi"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def echo(msg):
            return f"Echo: {msg}"

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if captured_chat.messages[-1]["role"] == "user" and "tool: echo" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "Echo", tools=[echo])

        expected_feedback = _format_tool_feedback(
            name="echo",
            input_str='{"msg": "hi"}',
            output_str="Echo: hi",
        )

        # Find the feedback message
        found = False
        for msg in captured_chat.messages:
            if expected_feedback in str(msg.get("content", "")):
                found = True
                break

        assert found, f"Expected feedback format not found in chat messages:\n{expected_feedback}"


# ─── Slice M: Assistant tool_calls discarded ──────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2b, Characterization

class TestAssistantToolCallsDiscarded:
    """CURRENT BEHAVIOR: Assistant messages in chat have content but NO tool_calls field.

    After SDD refactor, assistant messages should preserve the tool_calls array.
    """

    def test_assistant_message_has_no_tool_calls_field(self):
        """After tool-call response, assistant message in chat has NO tool_calls."""
        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        tool_response = _make_response(content="Let me check...", tool_calls=[tool_call])
        final_response = _make_response(content="It's 18C.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18}

        captured_chat = None
        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            if captured_chat.messages[-1]["role"] == "user" and "tool: get_weather" in str(captured_chat.messages[-1]):
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "Weather?", tools=[get_weather])

        # Find the assistant message from the tool-call response
        assistant_msgs = [m for m in captured_chat.messages if m["role"] == "assistant"]

        # The first assistant message should have content but NO tool_calls
        if len(assistant_msgs) >= 1:
            first_assistant = assistant_msgs[0]
            assert "tool_calls" not in first_assistant, (
                f"CURRENT BEHAVIOR: assistant messages should NOT have tool_calls, "
                f"but found: {first_assistant}"
            )
            # Content IS preserved
            assert first_assistant.get("content") == "Let me check..."


# ─── Slice N: Combined feedback format ────────────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 2b, Characterization

class TestCombinedFeedbackFormat:
    """CURRENT BEHAVIOR: Multi-tool results joined with \\n\\n into SINGLE user message.

    After SDD refactor, each tool result should be a SEPARATE role:"tool" message.
    """

    def test_multi_tool_results_in_single_user_message(self):
        """Two tool calls produce ONE user message with both results."""
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
            # Check if this is the tool feedback message
            content = str(captured_chat.messages[-1].get("content", ""))
            if "tool: tool_a" in content and "tool: tool_b" in content:
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "test", tools=[tool_a, tool_b])

        # Count user messages that contain tool feedback
        tool_feedback_user_msgs = []
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            if msg["role"] == "user" and ("tool: tool_a" in content or "tool: tool_b" in content):
                tool_feedback_user_msgs.append(msg)

        # Should be exactly ONE user message containing BOTH tool results
        assert len(tool_feedback_user_msgs) == 1, (
            f"Expected 1 combined user message, found {len(tool_feedback_user_msgs)}"
        )

        combined_content = tool_feedback_user_msgs[0]["content"]
        assert "tool: tool_a" in combined_content
        assert "tool: tool_b" in combined_content
        # Results are joined with \n\n
        assert "\n\n" in combined_content

    def test_no_per_tool_messages(self):
        """CURRENT BEHAVIOR: There are NOT separate messages per tool_call_id."""
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
            content = str(captured_chat.messages[-1].get("content", ""))
            if "tool: tool_a" in content and "tool: tool_b" in content:
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)

        run_agentic(client, "test", tools=[tool_a, tool_b])

        # Count how many user messages contain "tool: tool_a" specifically
        msgs_with_a = [
            m for m in captured_chat.messages
            if m["role"] == "user" and "tool: tool_a" in str(m.get("content", ""))
        ]
        msgs_with_b = [
            m for m in captured_chat.messages
            if m["role"] == "user" and "tool: tool_b" in str(m.get("content", ""))
        ]

        # Both should be in the SAME message (count == 1 each, and they're the same message)
        assert len(msgs_with_a) == 1
        assert len(msgs_with_b) == 1
        assert msgs_with_a[0] is msgs_with_b[0], (
            "CURRENT BEHAVIOR: both tool results should be in the SAME user message"
        )


# ─── Internal helpers ──────────────────────────────────────────────────────

def _make_mock_client(responses: list):
    """Create a mock client that returns responses in sequence."""
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
