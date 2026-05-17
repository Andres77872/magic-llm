"""GeminiToolAdapter — deprecated Gemini-format compatibility shim.

Handles Google Gemini models with native functionDeclarations/functionResponse format.
Canonical agent loops use magic_llm.engine.tooling instead.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class GeminiToolAdapter(ToolAdapter):
    """ToolAdapter implementation for Google Gemini models.

    Serializes tool definitions to Gemini's functionDeclarations format
    using parametersJsonSchema (JSON Schema v7), deserializes tool calls
    from functionCall parts, and injects tool results as role="user"
    messages with functionResponse parts.

    CRITICAL: Gemini returns finishReason="STOP" on ALL turns including
    tool-call turns. is_finished() checks tool_calls presence, NOT
    finish_reason.
    """

    def serialize_tool_defs(self, tools: list[Any]) -> Any:
        """Convert tool definitions to Gemini functionDeclarations format.

        Args:
            tools: List of tool definitions (callables, dicts, or Pydantic models).

        Returns:
            List with single dict: [{"functionDeclarations": [...]}]
            or None if no tools.
        """
        if not tools:
            return None

        return map_request_tools("google", tools, tool_choice=None).tools

    def deserialize_tool_calls(
        self, response: ModelChatResponse
    ) -> list[CanonicalToolCall]:
        """Extract tool calls from a Gemini-format response.

        Uses response.tool_calls (already normalized by process_generate).
        Parses JSON arguments string into dict.

        Args:
            response: The LLM response containing tool calls.

        Returns:
            List of CanonicalToolCall objects with parsed arguments.
        """
        return extract_tool_calls(response)

    def serialize_tool_results(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Inject tool results as a SINGLE role="user" message with functionResponse parts.

        CRITICAL: Validates completeness first — ALL tool_call IDs from the last
        assistant message must have matching ToolResult. Missing any ID raises
        a ValueError.

        Each result produces one functionResponse part:
            {"functionResponse": {
                "id": tool_call_id,
                "name": tool_name,
                "response": {"output": content}      # success
                "response": {"error": content}       # is_error=True
            }}

        Args:
            results: The list of tool execution results.
            chat: The ModelChat to mutate.

        Raises:
            ValueError: If tool_call IDs are missing from the results.
        """
        append_tool_results("google", chat, results)

    def is_finished(self, response: ModelChatResponse) -> bool:
        """Return True if the response indicates the loop should stop.

        Gemini returns finishReason="STOP" on ALL turns, so we check
        tool_calls presence instead:
            return response.tool_calls is None or len(response.tool_calls) == 0
        """
        return response.tool_calls is None or len(response.tool_calls) == 0

    def extract_final_text(self, response: ModelChatResponse) -> str:
        """Extract the final text content from a response."""
        return response.content or ""

    def validate_pair_integrity(self, chat: ModelChat) -> bool:
        """Check that ALL functionCall IDs from the last assistant message have matching functionResponse results.

        For Gemini, tool results appear as role="user" messages with
        functionResponse parts, NOT as role="tool" messages.

        Args:
            chat: The ModelChat to validate.

        Returns:
            True if all functionCall IDs have matching functionResponse results, False otherwise.
        """
        return validate_tool_result_integrity("google", chat)

    def _validate_completeness(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Validate that ALL tool_call IDs from the last assistant message have matching results.

        Raises ValueError if any tool_call ID is missing from the results.

        Args:
            results: The tool results to validate.
            chat: The ModelChat containing the assistant message.

        Raises:
            ValueError: If tool_call IDs are missing.
        """
        # Compatibility-only validation; canonical loops inject via engine tooling.
        expected = _last_assistant_tool_call_ids(chat)
        actual = {result.tool_call_id for result in results if result.tool_call_id}
        missing = expected - actual
        if missing:
            raise ValueError(
                f"Incomplete tool results: missing tool_call_id(s): {', '.join(sorted(missing))}"
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
