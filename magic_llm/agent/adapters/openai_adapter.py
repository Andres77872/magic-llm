"""OpenAIToolAdapter — OpenAI-format tool serialization/deserialization.

Handles OpenAI, Azure OpenAI, OpenRouter, and all OpenAI-compatible providers.
"""

from __future__ import annotations

import json
from typing import Any

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.util.tools_mapping import map_to_openai, normalize_openai_tools


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
        openai_tools, _ = map_to_openai(tools, tool_choice=None)
        return openai_tools

    def deserialize_tool_calls(
        self, response: ModelChatResponse
    ) -> list[CanonicalToolCall]:
        """Extract tool calls from an OpenAI-format response.

        Args:
            response: The LLM response containing tool calls.

        Returns:
            List of CanonicalToolCall objects with parsed arguments.
        """
        raw_calls = response.tool_calls
        if not raw_calls:
            return []

        result = []
        for tc in raw_calls:
            if tc.function is None:
                continue
            # Parse arguments from JSON string to dict
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            result.append(
                CanonicalToolCall(
                    id=tc.id or "",
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
        return result

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
        for result in results:
            chat.add_tool_result(
                tool_call_id=result.tool_call_id or "",
                content=result.content,
                is_error=result.is_error,
            )

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
        # Collect all expected tool_call_ids from assistant messages
        expected_ids: set[str] = set()
        for msg in chat.messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        expected_ids.add(tc_id)

        if not expected_ids:
            return True

        # Collect all actual tool_call_ids from tool messages
        actual_ids: set[str] = set()
        for msg in chat.messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    actual_ids.add(tc_id)

        return expected_ids.issubset(actual_ids)
