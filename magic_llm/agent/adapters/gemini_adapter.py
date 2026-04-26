"""GeminiToolAdapter — Gemini-format tool serialization/deserialization.

Handles Google Gemini models with native functionDeclarations/functionResponse format.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.util.tools_mapping import map_to_gemini

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

        gemini_tools, _ = map_to_gemini(tools, tool_choice=None)
        if not gemini_tools:
            return None

        return [{"functionDeclarations": gemini_tools}]

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
        # Validate completeness before serialization
        self._validate_completeness(results, chat)

        # Build a single user message with all functionResponse parts
        parts = []
        for result in results:
            if result.is_error:
                response_payload = {"error": result.content}
            else:
                response_payload = {"output": result.content}

            parts.append({
                "functionResponse": {
                    "id": result.tool_call_id or "",
                    "name": result.name,
                    "response": response_payload,
                }
            })

        chat.add_tool_messages(
            [
                {
                    "role": "user",
                    "content": parts,
                }
            ]
        )

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
        # Find the last assistant message with tool_calls
        last_assistant_tool_calls = None
        for msg in reversed(chat.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant_tool_calls = msg["tool_calls"]
                break

        if not last_assistant_tool_calls:
            return True

        # Collect expected tool_call IDs
        expected_ids: set[str] = set()
        for tc in last_assistant_tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                expected_ids.add(tc_id)

        if not expected_ids:
            return True

        # Collect actual functionResponse IDs from user messages
        actual_ids: set[str] = set()
        for msg in chat.messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "functionResponse" in part:
                            fr_id = part["functionResponse"].get("id")
                            if fr_id:
                                actual_ids.add(fr_id)

        return expected_ids.issubset(actual_ids)

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
        # Find the last assistant message with tool_calls
        last_assistant_tool_calls = None
        for msg in reversed(chat.messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_assistant_tool_calls = msg["tool_calls"]
                break

        if not last_assistant_tool_calls:
            return

        # Collect expected tool_call IDs
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
                f"Incomplete tool results: missing tool_call_id(s): {missing_str}"
            )
