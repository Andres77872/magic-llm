"""AnthropicToolAdapter — deprecated Anthropic-format compatibility shim.

Handles Anthropic Claude models with strict tool_use/tool_result pairing.
Canonical agent loops use magic_llm.engine.tooling instead.
"""

from __future__ import annotations

from typing import Any

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.engine.tooling import (
    append_tool_results,
    extract_tool_calls,
    map_request_tools,
    validate_tool_result_integrity,
)


class AnthropicToolAdapter(ToolAdapter):
    """ToolAdapter implementation for Anthropic Claude models.

    Serializes tool definitions to Anthropic's {"name", "description", "input_schema"} format,
    deserializes tool calls from response content blocks of type "tool_use", and injects
    tool results as a SINGLE role="user" message with content=[tool_result blocks].

    CRITICAL: Validates that ALL tool_use IDs from the last assistant message have matching
    ToolResult blocks before serialization. Missing any ID raises an integrity error.
    """

    def serialize_tool_defs(self, tools: list[Any]) -> Any:
        """Convert tool definitions to Anthropic request format.

        Args:
            tools: List of tool definitions (callables, dicts, or Pydantic models).

        Returns:
            List of Anthropic-format tool definitions, or None if no tools.
        """
        if not tools:
            return None
        return map_request_tools("anthropic", tools, tool_choice=None).tools

    def deserialize_tool_calls(
        self, response: ModelChatResponse
    ) -> list[CanonicalToolCall]:
        """Extract tool calls from an Anthropic-format response.

        Since EngineAnthropic maps responses to OpenAI-style ModelChatResponse,
        we can use response.tool_calls directly (already normalized).

        Args:
            response: The LLM response containing tool calls.

        Returns:
            List of CanonicalToolCall objects with parsed arguments.
        """
        return extract_tool_calls(response)

    def serialize_tool_results(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Inject tool results as a SINGLE role="user" message with content blocks.

        CRITICAL: Validates completeness first — ALL tool_use IDs from the last
        assistant message must have matching ToolResult. Missing any ID raises
        a ValueError.

        Args:
            results: The list of tool execution results.
            chat: The ModelChat to mutate.

        Raises:
            ValueError: If tool_use IDs are missing from the results.
        """
        append_tool_results("anthropic", chat, results)

    def is_finished(self, response: ModelChatResponse) -> bool:
        """Return True if the response indicates the loop should stop.

        EngineAnthropic maps stop_reason to OpenAI finish_reason:
        - "tool_use" -> "tool_calls" (continue)
        - "end_turn" -> "stop" (done)
        - "stop_sequence" -> "stop" (done)
        """
        return response.finish_reason == "stop"

    def extract_final_text(self, response: ModelChatResponse) -> str:
        """Extract the final text content from a response."""
        return response.content or ""

    def validate_pair_integrity(self, chat: ModelChat) -> bool:
        """Check that ALL tool_use IDs from the last assistant message have matching results.

        This is stricter than OpenAI — Anthropic requires ALL tool_use blocks to have
        corresponding tool_result blocks.

        Args:
            chat: The ModelChat to validate.

        Returns:
            True if all tool_use IDs have matching results, False otherwise.
        """
        return validate_tool_result_integrity("anthropic", chat)

    def _validate_completeness(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Validate that ALL tool_use IDs from the last assistant message have matching results.

        Raises ValueError if any tool_use ID is missing from the results.

        Args:
            results: The tool results to validate.
            chat: The ModelChat containing the assistant message.

        Raises:
            ValueError: If tool_use IDs are missing.
        """
        # Compatibility-only validation; canonical loops inject via engine tooling.
        expected = _last_assistant_tool_call_ids(chat)
        actual = {result.tool_call_id for result in results if result.tool_call_id}
        missing = expected - actual
        if missing:
            raise ValueError(
                f"Incomplete tool results: missing tool_use_id(s): {', '.join(sorted(missing))}"
            )


def _last_assistant_tool_call_ids(chat: ModelChat) -> set[str]:
    for msg in reversed(chat.messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = set()
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    ids.add(tc_id)
            return ids
    return set()
