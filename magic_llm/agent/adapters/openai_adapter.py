"""OpenAIToolAdapter — deprecated OpenAI-format compatibility shim.

Handles OpenAI, Azure OpenAI, OpenRouter, and all OpenAI-compatible providers.
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


class OpenAIToolAdapter(ToolAdapter):
    """ToolAdapter implementation for OpenAI-format providers.

    Serializes tool definitions to OpenAI's {"type": "function", "function": {...}} format,
    deserializes tool calls from response.choices[0].message.tool_calls, and injects
    tool results as separate role="tool" messages.
    """

    def serialize_tool_defs(self, tools: list[Any]) -> Any:
        """Convert tool definitions to OpenAI request format.

        Args:
            tools: List of tool definitions (callables, dicts, or Pydantic models).

        Returns:
            List of OpenAI-format tool definitions, or None if no tools.
        """
        if not tools:
            return None
        return map_request_tools("openai", tools, tool_choice=None).tools

    def deserialize_tool_calls(
        self, response: ModelChatResponse
    ) -> list[CanonicalToolCall]:
        """Extract tool calls from an OpenAI-format response.

        Args:
            response: The LLM response containing tool calls.

        Returns:
            List of CanonicalToolCall objects with parsed arguments.
        """
        return extract_tool_calls(response)

    def serialize_tool_results(
        self, results: list[ToolResult], chat: ModelChat
    ) -> None:
        """Inject tool results as separate role="tool" messages.

        Each ToolResult produces exactly one role="tool" message with
        matching tool_call_id.

        Args:
            results: The list of tool execution results.
            chat: The ModelChat to mutate.
        """
        append_tool_results("openai", chat, results)

    def is_finished(self, response: ModelChatResponse) -> bool:
        """Return True if the response indicates the loop should stop.

        OpenAI uses finish_reason="stop" to indicate completion.
        finish_reason="tool_calls" means more tool execution is needed.
        """
        return response.finish_reason == "stop"

    def extract_final_text(self, response: ModelChatResponse) -> str:
        """Extract the final text content from a response."""
        return response.content or ""

    def validate_pair_integrity(self, chat: ModelChat) -> bool:
        """Check that all tool calls have matching tool result messages.

        Scans assistant messages for tool_calls and verifies each tool_call_id
        has a corresponding role="tool" message.

        Args:
            chat: The ModelChat to validate.

        Returns:
            True if all tool calls have matching results, False otherwise.
        """
        return validate_tool_result_integrity("openai", chat)
