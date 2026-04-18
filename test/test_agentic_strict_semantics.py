"""Strict semantics tests for the agentic loop — TARGET behavior (xfail today).

TDD: agentic-tooling-flow-tests — Phase 3 (Strict Semantics)

These tests assert the DESIRED function-calling behavior per the SDD spec
(react-like-agent-loop-refactor). They are marked xfail because the current
run_agentic() implementation does NOT satisfy this contract.

When the SDD refactor lands (AgentLoop with ToolAdapter), these tests become
the acceptance criteria. Remove xfail markers once the production code matches.

┌──────────────────────────────────────────────────────────────────────────┐
│  DESIGN PRINCIPLE: Provider-Agnostic Canonical Contract                  │
│                                                                          │
│  These tests express a UNIFIED agent/tooling contract independent of     │
│  provider specifics. The contract is:                                    │
│                                                                          │
│  1. Assistant tool_calls are PRESERVED structurally in history.          │
│  2. Tool results are reinserted with CORRELATION IDs (not plain text).   │
│  3. Each tool result maps to its originating tool_call via ID.           │
│  4. Stop/continue is driven by NORMALIZED tool-call detection.           │
│                                                                          │
│  Provider-specific wire formats (OpenAI role:tool, Anthropic content     │
│  blocks, etc.) are handled by per-provider adapters. The agent loop      │
│  itself operates on the canonical contract.                              │
│                                                                          │
│  Adapter seam: ModelChat.add_tool_result() and                          │
│  ModelChat.add_tool_call_message() are the injection points.             │
│  Adapters translate between canonical and provider formats.              │
└──────────────────────────────────────────────────────────────────────────┘

Covers:
- Slice P: Tool result should be role:tool with tool_call_id
- Slice Q: Assistant should preserve tool_calls array
- Slice R: Per-tool correlation (separate messages per tool_call_id)
- Slice S: finish_reason awareness for stop/continue
- Slice P2: Canonical tool-result contract maps to provider adapters
- Slice S2: Stop/continue driven by normalized tool-call detection
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import run_agentic
from magic_llm.model.ModelChat import ModelChat
from magic_llm.model.ModelChatResponse import (
    ModelChatResponse, Choice, Message, UsageModel,
    ToolCall, FunctionCall,
)
from magic_llm.agent.types import (
    ToolResult, CanonicalToolCall, AgentBudget, AgentState,
)


# ─── Helper builders (shared across test classes) ─────────────────────────

def _make_mock_client(responses: list):
    """Create a mock client that returns responses in sequence."""
    client = MagicMock()
    client.llm.generate = MagicMock(side_effect=responses)
    return client


def _make_response(
    content=None,
    tool_calls=None,
    finish_reason="stop",
    model="test-model",
):
    """Build a valid ModelChatResponse."""
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


def _make_tool_call(
    id="call_1",
    name="get_weather",
    arguments='{"city":"London"}',
):
    """Build a valid ToolCall."""
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))


# ─── Slice P: Tool result should be role:tool with tool_call_id ───────────

class TestToolResultRoleAndCorrelation:
    """TARGET: Tool results are injected as role:"tool" with tool_call_id.

    Current behavior: role:"user" with plain text _format_tool_feedback().
    Target behavior: role:"tool" with tool_call_id matching the original call.

    This is the core semantic that enables provider adapters to serialize
    results correctly — OpenAI uses role:tool, Anthropic uses user+content
    blocks, but the CANONICAL contract is always correlated by tool_call_id.
    """

    @pytest.mark.xfail(
        reason="Current run_agentic injects tool results as role:user plain text, "
               "not role:tool with tool_call_id. Fixed by SDD refactor AgentLoop.",
        strict=True,
    )
    def test_tool_result_has_role_tool_not_user(self):
        """After tool execution, chat.messages contains role='tool' messages."""
        tool_call = _make_tool_call(id="call_weather", name="get_weather",
                                    arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="It's 18C in London.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18, "unit": "C"}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            # Detect tool result message to return final response
            for msg in captured_chat.messages:
                if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_weather":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather in London?", tools=[get_weather])

        # Find the tool result message
        tool_result_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) >= 1, (
            "Expected at least one role:'tool' message in chat history, "
            f"but found {len(tool_result_msgs)}. "
            "Tool results must use role:'tool' for provider adapters to serialize correctly."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic does not preserve tool_call_id correlation. "
               "Tool results are plain text with no structured ID.",
        strict=True,
    )
    def test_tool_result_has_matching_tool_call_id(self):
        """Tool result message has tool_call_id matching the original tool call."""
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
                if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_abc123":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # Find the tool result with matching ID
        matching = [
            m for m in captured_chat.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_abc123"
        ]
        assert len(matching) == 1, (
            "Expected exactly one role:'tool' message with tool_call_id='call_abc123', "
            f"but found {len(matching)}. "
            "Each tool result must correlate to its originating tool_call via ID."
        )
        # Content should be the tool output, not the _format_tool_feedback text
        assert "tool:" not in matching[0].get("content", ""), (
            "Tool result content should be the raw tool output, "
            "not the legacy _format_tool_feedback text format."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic stores tool results as plain text in role:user. "
               "The is_error flag should be preserved on role:tool messages.",
        strict=True,
    )
    def test_tool_result_preserves_error_flag(self):
        """Error tool results preserve is_error flag on the message."""
        tool_call = _make_tool_call(id="call_fail", name="boom_tool", arguments='{}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Recovered", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def boom_tool():
            raise ValueError("boom")

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_fail":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "test", tools=[boom_tool])

        error_msgs = [
            m for m in captured_chat.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_fail"
        ]
        assert len(error_msgs) == 1
        assert error_msgs[0].get("is_error") is True, (
            "Error tool results must have is_error=True flag for downstream consumers."
        )


# ─── Slice Q: Assistant should preserve tool_calls array ──────────────────

class TestAssistantToolCallPreservation:
    """TARGET: Assistant messages preserve the tool_calls array in history.

    Current behavior: Only text content is added via chat.add_assistant_message(),
    tool_calls array is discarded entirely.

    Target behavior: chat.add_tool_call_message(tool_calls, content) is called,
    preserving the full tool_calls array for provider adapters and integrity checks.
    """

    @pytest.mark.xfail(
        reason="Current run_agentic discards tool_calls from assistant messages. "
               "Only text content is preserved via add_assistant_message().",
        strict=True,
    )
    def test_assistant_message_preserves_tool_calls_array(self):
        """After tool-call response, assistant message in chat includes tool_calls."""
        tool_call = _make_tool_call(id="call_1", name="get_weather",
                                    arguments='{"city":"London"}')
        tool_response = _make_response(content="Let me check...", tool_calls=[tool_call])
        final_response = _make_response(content="It's 18C.", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            for msg in captured_chat.messages:
                if msg.get("role") == "tool":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # Find assistant messages with tool_calls
        assistant_with_tools = [
            m for m in captured_chat.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_with_tools) >= 1, (
            "Expected at least one assistant message with tool_calls preserved, "
            f"but found {len(assistant_with_tools)}. "
            "Assistant tool_calls must be preserved for provider adapters and "
            "integrity validation (validate_tool_integrity)."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic does not preserve tool_call IDs in assistant messages, "
               "making it impossible to correlate tool results with their originating calls.",
        strict=True,
    )
    def test_assistant_tool_calls_have_original_ids(self):
        """Preserved tool_calls retain their original provider IDs."""
        tool_call = _make_tool_call(id="call_xyz789", name="get_weather",
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
                if msg.get("role") == "tool":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # Find the assistant message with tool_calls
        assistant_msgs = [
            m for m in captured_chat.messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert len(assistant_msgs) >= 1

        # Check that the tool_call ID is preserved
        preserved_ids = []
        for msg in assistant_msgs:
            for tc in msg.get("tool_calls", []):
                # tool_calls may be list of ToolCall objects or dicts
                if hasattr(tc, "id"):
                    preserved_ids.append(tc.id)
                elif isinstance(tc, dict):
                    preserved_ids.append(tc.get("id"))

        assert "call_xyz789" in preserved_ids, (
            f"Expected tool_call_id 'call_xyz789' to be preserved in assistant message, "
            f"but found IDs: {preserved_ids}. "
            "Provider IDs must be preserved for per-tool correlation."
        )


# ─── Slice R: Per-tool correlation ────────────────────────────────────────

class TestPerToolCorrelation:
    """TARGET: Each tool result is a SEPARATE message with its own tool_call_id.

    Current behavior: All tool results from one iteration are COMBINED into
    a single role:user message with \n\n separator.

    Target behavior: Each tool result produces its own role:tool message,
    enabling per-tool correlation and parallel tool result serialization
    by provider adapters (OpenAI: N role:tool messages, Anthropic: 1 user
    message with N content blocks).
    """

    @pytest.mark.xfail(
        reason="Current run_agentic combines all tool results into ONE role:user message. "
               "Target: separate role:tool messages per tool_call_id.",
        strict=True,
    )
    def test_multi_tool_produces_separate_tool_messages(self):
        """Two tool calls produce TWO separate role:tool messages, not one combined."""
        call_a = _make_tool_call(id="call_a", name="tool_a", arguments='{}')
        call_b = _make_tool_call(id="call_b", name="tool_b", arguments='{}')
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
            tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
            if len(tool_msgs) >= 2:
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "test", tools=[tool_a, tool_b])

        # Count role:tool messages
        tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2, (
            f"Expected 2 separate role:'tool' messages (one per tool_call), "
            f"but found {len(tool_msgs)}. "
            "Each tool result must be a separate message for per-tool correlation."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic has no per-tool correlation — all results "
               "are in one combined message with no tool_call_id references.",
        strict=True,
    )
    def test_each_tool_message_has_unique_tool_call_id(self):
        """Each role:tool message has a unique tool_call_id matching its call."""
        call_a = _make_tool_call(id="call_a", name="tool_a", arguments='{}')
        call_b = _make_tool_call(id="call_b", name="tool_b", arguments='{}')
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
            tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
            if len(tool_msgs) >= 2:
                return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "test", tools=[tool_a, tool_b])

        tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        tool_call_ids = {m.get("tool_call_id") for m in tool_msgs}

        assert tool_call_ids == {"call_a", "call_b"}, (
            f"Expected tool_call_ids {{'call_a', 'call_b'}}, but found {tool_call_ids}. "
            "Each tool result message must have a unique tool_call_id for correlation."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic combines results with \\n\\n, losing per-tool identity. "
               "Provider adapters need individual ToolResult objects to serialize correctly.",
        strict=True,
    )
    def test_tool_results_not_combined_with_newline_separator(self):
        """Tool results are NOT joined with \\n\\n into a single message."""
        call_a = _make_tool_call(id="call_a", name="tool_a", arguments='{}')
        call_b = _make_tool_call(id="call_b", name="tool_b", arguments='{}')
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
            # Detect combined feedback (current) or separate tool messages (target)
            for msg in captured_chat.messages:
                content = str(msg.get("content", ""))
                # Target: two separate role:tool messages
                tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
                if len(tool_msgs) >= 2:
                    return final_response
                # Current: single role:user with both results
                if "tool: tool_a" in content and "tool: tool_b" in content:
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "test", tools=[tool_a, tool_b])

        # Verify no single message contains BOTH tool results
        # (In target behavior, each tool result is in its own message)
        for msg in captured_chat.messages:
            content = str(msg.get("content", ""))
            # Check any message that carries tool feedback
            if "tool: tool_a" in content or "tool: tool_b" in content:
                assert not ("result_a" in content and "result_b" in content), (
                    "A single message contains results from multiple tools. "
                    "Each tool result must be in its own message for per-tool correlation."
                )


# ─── Slice S: finish_reason awareness ─────────────────────────────────────

class TestFinishReasonAwareness:
    """TARGET: Loop continuation is driven by normalized tool-call detection.

    Current behavior: Loop only checks `response.tool_calls` list.
    `finish_reason` is logged but NOT used for termination decisions.

    Target behavior: The adapter's `is_finished(response)` method determines
    continuation. For OpenAI, finish_reason="tool_calls" means continue even
    if tool_calls list is empty (edge case). finish_reason="stop" means done.

    This ensures the agent loop doesn't leak provider-specific logic — the
    adapter normalizes the signal, the loop acts on it.
    """

    @pytest.mark.xfail(
        reason="Current run_agentic ignores finish_reason for termination. "
               "Only checks tool_calls list presence. finish_reason='tool_calls' "
               "with empty list would incorrectly break the loop.",
        strict=True,
    )
    def test_finish_reason_tool_calls_means_continue(self):
        """finish_reason='tool_calls' signals continuation even with empty tool_calls list."""
        # Edge case: model signals tool_calls needed but list is empty
        # (can happen with some providers/models)
        response = _make_response(
            content=None,
            tool_calls=[],  # Empty list
            finish_reason="tool_calls",  # But finish_reason says tools needed
        )
        final_response = _make_response(content="Final answer", finish_reason="stop")
        client = _make_mock_client([response, final_response])

        # No tools registered — if loop breaks, we never get to final_response
        result = run_agentic(client, "test", tools=[], max_iterations=5)

        # With finish_reason awareness, the loop should continue (adapter says not finished)
        # and make a second generate call
        assert client.llm.generate.call_count >= 2, (
            f"Expected at least 2 generate calls (finish_reason='tool_calls' should "
            f"signal continuation), but got {client.llm.generate.call_count}. "
            "Loop should use adapter.is_finished() not just tool_calls list check."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic does not use adapter.is_finished(). "
               "finish_reason='stop' with tool_calls present is not handled.",
        strict=True,
    )
    def test_finish_reason_stop_means_done_even_with_tool_calls(self):
        """finish_reason='stop' signals completion even if tool_calls list exists.

        Some providers may return tool_calls metadata with finish_reason='stop'
        (e.g., parallel tool calls where some completed). The adapter should
        interpret this as 'done'.
        """
        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        response = _make_response(
            content="I'll check the weather.",
            tool_calls=[tool_call],
            finish_reason="stop",  # Provider says done despite tool_calls
        )
        client = _make_mock_client([response])

        def get_weather(city):
            return {"temp": 18}

        result = run_agentic(client, "Weather?", tools=[get_weather], max_iterations=5)

        # With finish_reason awareness, adapter.is_finished() would return True,
        # so only 1 generate call should be made
        assert client.llm.generate.call_count == 1, (
            f"Expected 1 generate call (finish_reason='stop' should signal done), "
            f"but got {client.llm.generate.call_count}. "
            "Loop should use adapter.is_finished() to determine completion."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic has no adapter abstraction. "
               "Tool-call detection is hardcoded to response.tool_calls check.",
        strict=True,
    )
    def test_loop_uses_normalized_tool_detection_not_provider_leakage(self):
        """Loop continuation is driven by adapter signal, not raw tool_calls check.

        This test documents the expected architecture: the agent loop should
        call adapter.is_finished(response) and adapter.deserialize_tool_calls(response)
        rather than directly accessing response.tool_calls. This prevents
        provider-specific leakage in the loop logic.
        """
        # This is an architecture documentation test.
        # It verifies that the loop uses normalized signals.
        # Currently it FAILS because the loop directly checks response.tool_calls.

        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
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
                if msg.get("role") == "tool":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # The assertion here is structural: after refactor, tool results
        # should be in role:tool messages (normalized format), not role:user.
        # This proves the loop used the adapter to serialize results.
        tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1, (
            "Expected role:'tool' messages, proving the loop uses adapter "
            "serialization rather than raw _format_tool_feedback injection."
        )


# ─── Slice P2: Canonical tool-result contract → provider adapters ─────────

class TestCanonicalToolResultContract:
    """TARGET: The canonical ToolResult model can map to provider adapters
    without changing agent-loop semantics.

    These tests validate the TYPE CONTRACT that adapters depend on.
    The ToolResult model (from magic_llm.agent.types) and ModelChat methods
    (add_tool_result, add_tool_call_message) form the canonical seam.

    These tests should PASS today because the types and ModelChat methods
    already exist from SDD Phase 1. They document the contract that the
    agent loop must use.
    """

    def test_tool_result_model_has_required_fields(self):
        """ToolResult model has all fields needed for provider adapter mapping."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            content='{"temp": 18}',
            is_error=False,
        )
        assert result.tool_call_id == "call_123"
        assert result.name == "get_weather"
        assert result.content == '{"temp": 18}'
        assert result.is_error is False
        assert result.error is None
        assert result.error_type is None
        assert result.duration_ms == 0.0
        assert result.is_deduplicated is False

    def test_tool_result_error_fields(self):
        """ToolResult captures error metadata for structured error injection."""
        result = ToolResult(
            tool_call_id="call_err",
            name="boom_tool",
            content='{"error": "boom", "type": "ValueError"}',
            is_error=True,
            error="boom",
            error_type="ValueError",
            duration_ms=42.5,
        )
        assert result.is_error is True
        assert result.error == "boom"
        assert result.error_type == "ValueError"
        assert result.duration_ms == 42.5

    def test_canonical_tool_call_is_provider_agnostic(self):
        """CanonicalToolCall represents a tool call without provider specifics."""
        call = CanonicalToolCall(
            id="call_abc",
            name="get_weather",
            arguments={"city": "London", "unit": "celsius"},
        )
        assert call.id == "call_abc"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "London", "unit": "celsius"}
        # Arguments are always parsed dicts, never JSON strings
        assert isinstance(call.arguments, dict)

    def test_canonical_tool_call_equality_and_hash(self):
        """CanonicalToolCall supports deduplication via equality/hashing."""
        call_a = CanonicalToolCall(id="call_1", name="fn", arguments={"x": 1})
        call_b = CanonicalToolCall(id="call_1", name="fn", arguments={"x": 1})
        call_c = CanonicalToolCall(id="call_2", name="fn", arguments={"x": 1})

        assert call_a == call_b
        assert call_a != call_c
        assert hash(call_a) == hash(call_b)

    def test_model_chat_add_tool_result_appends_tool_role(self):
        """ModelChat.add_tool_result() appends a role:tool message.

        This is the CANONICAL injection point. Provider adapters call this
        (or add_tool_messages) to inject results. The agent loop should
        use this, not add_user_message with formatted text.
        """
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_result(
            tool_call_id="call_123",
            content='{"temp": 18}',
            is_error=False,
        )

        assert len(chat.messages) == 2
        tool_msg = chat.messages[1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["content"] == '{"temp": 18}'
        assert tool_msg["is_error"] is False

    def test_model_chat_add_tool_result_error(self):
        """ModelChat.add_tool_result() preserves error flag."""
        chat = ModelChat()
        chat.add_tool_result(
            tool_call_id="call_err",
            content='{"error": "boom"}',
            is_error=True,
        )

        tool_msg = chat.messages[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_err"
        assert tool_msg["is_error"] is True

    def test_model_chat_add_tool_call_message_preserves_tool_calls(self):
        """ModelChat.add_tool_call_message() preserves tool_calls array.

        This is the CANONICAL injection point for assistant tool-call messages.
        The agent loop should use this, not add_assistant_message which discards
        tool_calls.
        """
        chat = ModelChat()
        tool_calls_data = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
            }
        ]
        chat.add_tool_call_message(
            tool_calls=tool_calls_data,
            content="Let me check...",
        )

        assert len(chat.messages) == 1
        assistant_msg = chat.messages[0]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check..."
        assert assistant_msg["tool_calls"] == tool_calls_data

    def test_model_chat_add_tool_messages_batch_injection(self):
        """ModelChat.add_tool_messages() supports batch injection for adapters.

        Provider adapters (e.g., Anthropic) may need to inject multiple messages
        at once (e.g., a single user message with multiple content blocks).
        This method provides the seam for that.
        """
        chat = ModelChat()
        chat.add_user_message("Hello")

        # Simulate what an adapter might inject
        adapter_messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result_1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result_2"},
        ]
        chat.add_tool_messages(adapter_messages)

        assert len(chat.messages) == 3
        assert chat.messages[1]["tool_call_id"] == "call_1"
        assert chat.messages[2]["tool_call_id"] == "call_2"

    def test_tool_result_to_model_chat_message_mapping(self):
        """A ToolResult can be mapped to a ModelChat message dict.

        This documents the expected adapter contract: given a ToolResult,
        an adapter produces one or more message dicts to inject into ModelChat.
        """
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            content='{"temp": 18}',
            is_error=False,
        )

        # The canonical mapping (OpenAI-style):
        message = {
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }

        chat = ModelChat()
        chat.add_tool_messages([message])

        assert chat.messages[0]["role"] == "tool"
        assert chat.messages[0]["tool_call_id"] == "call_123"
        assert chat.messages[0]["content"] == '{"temp": 18}'


# ─── Slice S2: Stop/continue driven by normalized detection ───────────────

class TestNormalizedStopContinueBehavior:
    """TARGET: Stop/continue behavior is driven by normalized tool-call
    detection, not provider-specific leakage in the loop.

    These tests document the expected architecture: the loop should use
    adapter.is_finished() and adapter.deserialize_tool_calls() rather than
    directly accessing provider-specific response fields.

    Since the adapter infrastructure doesn't exist yet in the loop, these
    are xfail. They become green when AgentLoop replaces run_agentic().
    """

    @pytest.mark.xfail(
        reason="Current run_agentic directly checks response.tool_calls. "
               "Target: loop uses adapter.is_finished() for termination signal.",
        strict=True,
    )
    def test_loop_termination_uses_adapter_signal(self):
        """Loop termination should use adapter.is_finished(), not raw tool_calls check.

        This test documents the target architecture: the agent loop should
        delegate termination decisions to the adapter. This allows different
        providers to signal completion differently (OpenAI: finish_reason,
        Anthropic: stop_reason) without changing the loop logic.
        """
        # The test passes when the loop uses adapter.is_finished() internally.
        # Currently it fails because the loop checks `if not tool_calls: break` directly.

        tool_call = _make_tool_call(name="get_weather", arguments='{"city":"London"}')
        tool_response = _make_response(content=None, tool_calls=[tool_call])
        final_response = _make_response(content="Done", finish_reason="stop")
        client = _make_mock_client([tool_response, final_response])

        def get_weather(city):
            return {"temp": 18}

        captured_chat = None

        def capture_chat(chat, **kwargs):
            nonlocal captured_chat
            captured_chat = chat
            # If tool results are injected as role:tool, adapter was used
            for msg in captured_chat.messages:
                if msg.get("role") == "tool":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        run_agentic(client, "Weather?", tools=[get_weather])

        # Verify normalized message format (proves adapter was used)
        tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1, (
            "Expected role:'tool' messages, proving the loop uses adapter "
            "for tool result serialization."
        )

    @pytest.mark.xfail(
        reason="Current run_agentic has no adapter abstraction for tool call "
               "deserialization. Directly uses response.tool_calls property.",
        strict=True,
    )
    def test_tool_extraction_uses_adapter_deserialize(self):
        """Tool call extraction should use adapter.deserialize_tool_calls().

        This documents the target architecture: the loop should not directly
        access response.tool_calls. Instead, it should call the adapter's
        deserialize method, which handles provider-specific extraction
        (OpenAI: response.choices[0].message.tool_calls,
         Anthropic: response.content blocks of type 'tool_use').
        """
        # After refactor, this is verified by the fact that the loop
        # correctly handles tool calls regardless of provider format.
        # For now, we document the expected behavior.

        tool_call = _make_tool_call(id="call_1", name="get_weather",
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
                if msg.get("role") == "tool":
                    return final_response
            return tool_response

        client.llm.generate = MagicMock(side_effect=capture_chat)
        result = run_agentic(client, "Weather?", tools=[get_weather])

        # Verify tool was executed correctly (proves extraction worked)
        assert result.content == "Done"

        # Verify normalized tool result format
        tool_msgs = [m for m in captured_chat.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1


# ─── Integration: Canonical Contract + Adapter Seam ───────────────────────

class TestCanonicalContractAdapterSeam:
    """Integration tests for the canonical contract + adapter seam.

    These tests verify that the canonical types (ToolResult, CanonicalToolCall)
    and ModelChat methods form a coherent seam that provider adapters can
    build on. They should PASS today since the types exist.
    """

    def test_openai_style_adapter_can_serialize_tool_results(self):
        """An OpenAI-style adapter can serialize ToolResults as role:tool messages.

        This documents the expected adapter behavior WITHOUT implementing the adapter.
        It shows how the canonical contract maps to provider wire format.
        """
        results = [
            ToolResult(tool_call_id="call_1", name="tool_a", content="result_a"),
            ToolResult(tool_call_id="call_2", name="tool_b", content="result_b"),
        ]

        # Simulate what OpenAIToolAdapter.serialize_tool_results would do:
        chat = ModelChat()
        for result in results:
            chat.add_tool_result(
                tool_call_id=result.tool_call_id,
                content=result.content,
                is_error=result.is_error,
            )

        assert len(chat.messages) == 2
        assert chat.messages[0]["role"] == "tool"
        assert chat.messages[0]["tool_call_id"] == "call_1"
        assert chat.messages[1]["role"] == "tool"
        assert chat.messages[1]["tool_call_id"] == "call_2"

    def test_anthropic_style_adapter_can_bundle_tool_results(self):
        """An Anthropic-style adapter can bundle ToolResults in a single user message.

        Anthropic requires tool results as a single user message with content
        blocks. This documents how the canonical contract maps to that format.
        """
        results = [
            ToolResult(tool_call_id="toolu_1", name="tool_a", content="result_a"),
            ToolResult(tool_call_id="toolu_2", name="tool_b", content="result_b"),
        ]

        # Simulate what AnthropicToolAdapter.serialize_tool_results would do:
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
        assert msg["content"][0]["tool_use_id"] == "toolu_1"
        assert msg["content"][1]["tool_use_id"] == "toolu_2"

    @pytest.mark.xfail(
        reason="ModelChat.validate_tool_integrity() not yet implemented. "
               "Defined in SDD spec §ModelChat.validate_tool_integrity() — part of the refactor.",
        strict=True,
    )
    def test_tool_integrity_validation_detects_orphaned_calls(self):
        """ModelChat.validate_tool_integrity() detects unmatched tool_call/tool_result pairs.

        This validates the integrity check that adapters rely on before
        serializing results.
        """
        from magic_llm.exception.ChatException import ChatException

        chat = ModelChat()
        chat.add_user_message("Hello")

        # Add assistant message with 2 tool_calls
        chat.add_tool_call_message(tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ])

        # Add only 1 tool result (orphaned call_2)
        chat.add_tool_result(tool_call_id="call_1", content="result_1")

        # Integrity check should fail
        with pytest.raises(ChatException) as exc_info:
            chat.validate_tool_integrity()

        assert exc_info.value.error_code == "TOOL_INTEGRITY_ERROR"
        assert "call_2" in str(exc_info.value)

    @pytest.mark.xfail(
        reason="ModelChat.validate_tool_integrity() not yet implemented. "
               "Defined in SDD spec §ModelChat.validate_tool_integrity() — part of the refactor.",
        strict=True,
    )
    def test_tool_integrity_validation_passes_with_matched_pairs(self):
        """ModelChat.validate_tool_integrity() returns True when all pairs match."""
        chat = ModelChat()
        chat.add_user_message("Hello")
        chat.add_tool_call_message(tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
        ])
        chat.add_tool_result(tool_call_id="call_1", content="result_1")

        # Should pass
        assert chat.validate_tool_integrity() is True

    def test_agent_budget_and_state_are_provider_agnostic(self):
        """AgentBudget and AgentState have no provider-specific fields.

        These types are part of the canonical contract — they work the same
        regardless of which provider adapter is used.
        """
        budget = AgentBudget(max_iterations=5, max_output_tokens=1000)
        assert budget.max_iterations == 5
        assert budget.max_output_tokens == 1000
        assert budget.max_input_tokens is None
        assert budget.wall_clock_timeout is None

        state = AgentState(step=2, total_output_tokens=500)
        assert state.step == 2
        assert state.total_output_tokens == 500
        assert state.messages == []
        assert state.executed_fingerprints == set()
