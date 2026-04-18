"""AnthropicToolAdapter — Anthropic-format tool serialization/deserialization.

Handles Anthropic Claude models with strict tool_use/tool_result pairing.
"""

from __future__ import annotations

import json
from typing import Any

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.util.tools_mapping import map_to_anthropic, normalize_openai_tools


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
        anthropic_tools, _ = map_to_anthropic(tools, tool_choice=None)
        return anthropic_tools

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
        raw_calls = response.tool_calls
        if not raw_calls:
            return []

        result = []
        for tc in raw_calls:
            if tc.function is None:
                continue
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
        # Validate completeness before serialization
        self._validate_completeness(results, chat)

        # Build a single user message with all tool_result content blocks
        content_blocks = []
        for result in results:
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id or "",
                    "content": result.content,
                }
            )

        chat.add_tool_messages(
            [
                {
                    "role": "user",
                    "content": content_blocks,
                }
            ]
        )

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
        # Find the last assistant message with tool_calls
        last_assistant_tool_calls = None
        for msg in reversed(chat.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant_tool_calls = msg["tool_calls"]
                break

        if not last_assistant_tool_calls:
            return True

        # Collect expected tool_use IDs
        expected_ids: set[str] = set()
        for tc in last_assistant_tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                expected_ids.add(tc_id)

        if not expected_ids:
            return True

        # Collect actual tool_call_ids from tool messages
        actual_ids: set[str] = set()
        for msg in chat.messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    actual_ids.add(tc_id)

        return expected_ids.issubset(actual_ids)

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
        # Find the last assistant message with tool_calls
        last_assistant_tool_calls = None
        for msg in reversed(chat.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant_tool_calls = msg["tool_calls"]
                break

        if not last_assistant_tool_calls:
            return

        # Collect expected tool_use IDs
        expected_ids: set[str] = set()
        for tc in last_assistant_tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                expected_ids.add(tc_id)

        if not expected_ids:
            return

        # Collect actual tool_call_ids from results
        actual_ids: set[str] = set()
        for result in results:
            if result.tool_call_id:
                actual_ids.add(result.tool_call_id)

        # Check for missing IDs
        missing = expected_ids - actual_ids
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Incomplete tool results: missing tool_use_id(s): {missing_str}"
            )
