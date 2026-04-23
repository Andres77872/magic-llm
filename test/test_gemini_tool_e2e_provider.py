"""Provider-functional E2E test for native Gemini tool calling.

Uses the REAL Gemini API (gemini-3.1-pro-preview) to validate the full
round-trip: model emits tool call -> local tool executes -> tool result
sent back -> model returns final text.

Skipped when no Google API key is available.
"""

import json
import os

import pytest

from conftest import resolve_keys_file

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.types import AgentBudget

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Resolve keys file
_KEYS_FILE = resolve_keys_file()
with open(_KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

# Skip entirely if no Google key
if "google" not in ALL_KEYS:
    pytest.skip("No Google API key in keys file", allow_module_level=True)

GOOGLE_KEYS = dict(ALL_KEYS["google"])
MODEL = "gemini-3.1-pro-preview"


class TestGeminiNativeToolCallingE2E:
    """Real E2E validation of native Gemini tool calling round-trip."""

    def test_single_tool_call_round_trip(self):
        """Full round-trip: user prompt -> tool call -> tool exec -> final text.

        Manually orchestrates the two-turn flow to prove the complete E2E:
        1. First generate: model emits tool call
        2. Execute tool locally
        3. Serialize tool result into chat
        4. Second generate: model returns final text using tool result
        """

        def get_favorite_color(person: str) -> str:
            """Return the favorite color of a person."""
            return "blue"

        google_keys = dict(GOOGLE_KEYS)
        google_keys.pop("engine", None)
        client = MagicLLM(engine="google", model=MODEL, **google_keys)

        from magic_llm.agent.adapters import GeminiToolAdapter
        from magic_llm.agent.types import ToolResult, CanonicalToolCall
        import json

        adapter = GeminiToolAdapter()

        # --- Turn 1: Force tool call ---
        chat = ModelChat()
        chat.add_user_message(
            "What is the favorite color of Alice? Use the get_favorite_color tool."
        )
        tool_defs = adapter.serialize_tool_defs([get_favorite_color])

        response1 = client.llm.generate(
            chat,
            tools=tool_defs,
            tool_choice="required",
        )

        # Validate tool call was emitted
        tool_calls = adapter.deserialize_tool_calls(response1)
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        assert tool_calls[0].name == "get_favorite_color"
        assert "person" in tool_calls[0].arguments

        # Add assistant message with tool call
        chat.add_assistant_message(response1.content)
        chat.add_tool_call_message(
            tool_calls=[{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            } for tc in tool_calls],
            content=response1.content,
        )

        # --- Execute tool locally ---
        tool_call = tool_calls[0]
        tool_result = get_favorite_color(**tool_call.arguments)
        results = [
            ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=tool_result,
                is_error=False,
            )
        ]

        # --- Inject tool result ---
        adapter.serialize_tool_results(results, chat)

        # --- Turn 2: Get final text (auto, no forced tool call) ---
        response2 = client.llm.generate(
            chat,
            tools=tool_defs,
            tool_choice="auto",
        )

        # Validate final response
        assert response2 is not None, "Expected final response"
        assert response2.content, "Expected non-empty final text"
        assert "blue" in response2.content.lower(), (
            f"Expected tool result 'blue' in final response, got: {response2.content}"
        )

    def test_tool_call_emits_function_call(self):
        """Validate that the first generate call returns a tool call (not just text).

        This proves the model actually used function calling, not just answered directly.
        """
        google_keys = dict(GOOGLE_KEYS)
        google_keys.pop("engine", None)

        def get_favorite_color(person: str) -> str:
            """Return the favorite color of a person."""
            return "blue"

        client = MagicLLM(engine="google", model=MODEL, **google_keys)

        chat = ModelChat()
        chat.add_user_message("What is the favorite color of Bob?")

        # First call: should return a tool call
        from magic_llm.agent.adapters import GeminiToolAdapter

        adapter = GeminiToolAdapter()
        tool_defs = adapter.serialize_tool_defs([get_favorite_color])

        response = client.llm.generate(
            chat,
            tools=tool_defs,
            tool_choice="auto",
        )

        # Validate response structure
        assert response is not None
        assert response.tool_calls is not None and len(response.tool_calls) > 0, (
            f"Expected tool calls, got content={response.content!r}, "
            f"tool_calls={response.tool_calls}"
        )

        tool_call = response.tool_calls[0]
        assert tool_call.function.name == "get_favorite_color"
        args = json.loads(tool_call.function.arguments)
        assert "person" in args, f"Expected 'person' argument, got: {args}"

    def test_tool_result_injection_format(self):
        """Validate that tool results are serialized as Gemini functionResponse parts."""
        from magic_llm.agent.adapters import GeminiToolAdapter
        from magic_llm.agent.types import ToolResult

        adapter = GeminiToolAdapter()
        chat = ModelChat()
        chat.add_user_message("test")
        chat.add_assistant_message("thinking...")
        chat.add_tool_call_message(
            tool_calls=[{
                "id": "call_test_1",
                "type": "function",
                "function": {
                    "name": "get_favorite_color",
                    "arguments": '{"person": "Alice"}',
                },
            }],
            content="thinking...",
        )

        results = [
            ToolResult(
                tool_call_id="call_test_1",
                name="get_favorite_color",
                content="blue",
                is_error=False,
            )
        ]

        adapter.serialize_tool_results(results, chat)

        # Find the functionResponse message
        fr_msg = None
        for msg in chat.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and "functionResponse" in part:
                        fr_msg = msg
                        break
            if fr_msg:
                break

        assert fr_msg is not None, "Expected a user message with functionResponse parts"
        fr_parts = [p for p in fr_msg["content"] if "functionResponse" in p]
        assert len(fr_parts) == 1
        assert fr_parts[0]["functionResponse"]["name"] == "get_favorite_color"
        assert fr_parts[0]["functionResponse"]["id"] == "call_test_1"
