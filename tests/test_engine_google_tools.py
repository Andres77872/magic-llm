"""Integration tests for EngineGoogle tool request serialization.

Tests cover:
- prepare_data_sync() with tools → correct functionDeclarations and toolConfig
- prepare_data_sync() without tools → no tool keys (backward compat)
- prepare_data_sync() with tools=None → no tool keys
- prepare_data() async variant with tools → same assertions
- process_generate() with native id → tool_call uses native ID
- process_generate() without id → tool_call uses synthetic ID
- process_generate() with missing args → arguments = "{}"
- process_generate() with missing name → part skipped (no crash)
- prepare_stream_response() with text part → delta.content = text
- prepare_stream_response() with functionCall part → delta.content = "", tool_calls populated
- prepare_stream_response() with mixed parts → delta.content = text, tool_calls populated
"""

import json
import time
from unittest.mock import patch, MagicMock

import pytest

from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_engine():
    """Create an EngineGoogle instance for testing."""
    return EngineGoogle(api_key="test-api-key", model="gemini-2.5-flash")


def _get_weather(city: str) -> str:
    """Get weather for a city."""
    return "sunny"


# ─── prepare_data_sync() with tools ────────────────────────────────────────


class TestPrepareDataSyncWithTools:
    """prepare_data_sync() with tools kwarg."""

    def test_tools_produces_function_declarations(self):
        """With tools → data["tools"] has correct functionDeclarations structure."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("What's the weather?")

        json_bytes, headers, data = engine.prepare_data_sync(chat, tools=[_get_weather], tool_choice="auto")

        assert "tools" in data
        assert len(data["tools"]) == 1
        assert "functionDeclarations" in data["tools"][0]

        decls = data["tools"][0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "_get_weather"
        assert "parametersJsonSchema" in decls[0]

    def test_tool_choice_produces_tool_config(self):
        """With tool_choice → data["toolConfig"] has correct functionCallingConfig."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("What's the weather?")

        json_bytes, headers, data = engine.prepare_data_sync(chat, tools=[_get_weather], tool_choice="auto")

        assert "toolConfig" in data
        assert data["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}

    def test_generation_config_no_tool_keys(self):
        """tools and tool_choice NOT inside generationConfig."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("What's the weather?")

        json_bytes, headers, data = engine.prepare_data_sync(chat, tools=[_get_weather], tool_choice="auto")

        assert "tools" not in data["generationConfig"]
        assert "tool_choice" not in data["generationConfig"]


# ─── prepare_data_sync() without tools (backward compat) ───────────────────


class TestPrepareDataSyncBackwardCompat:
    """prepare_data_sync() without tools → backward compatibility."""

    def test_no_tools_no_tool_keys(self):
        """Without tools → no "tools" or "toolConfig" keys in data."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("Hello")

        json_bytes, headers, data = engine.prepare_data_sync(chat)

        assert "tools" not in data
        assert "toolConfig" not in data

    def test_tools_none_no_tool_keys(self):
        """tools=None → no tool keys in data."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("Hello")

        json_bytes, headers, data = engine.prepare_data_sync(chat, tools=None)

        assert "tools" not in data
        assert "toolConfig" not in data

    def test_request_structure_unchanged(self):
        """Non-tool request has same keys as before: contents, generationConfig."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("Hello")

        json_bytes, headers, data = engine.prepare_data_sync(chat)

        assert "contents" in data
        assert "generationConfig" in data
        assert len(data["contents"]) == 1
        assert data["contents"][0]["role"] == "user"


# ─── prepare_data() async variant ──────────────────────────────────────────


class TestPrepareDataAsync:
    """prepare_data() async variant with tools."""

    @pytest.mark.asyncio
    async def test_async_tools_produces_function_declarations(self):
        """Async prepare_data with tools → same structure as sync."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("What's the weather?")

        json_bytes, headers, data = await engine.prepare_data(chat, tools=[_get_weather], tool_choice="required")

        assert "tools" in data
        assert "toolConfig" in data
        assert data["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}

    @pytest.mark.asyncio
    async def test_async_no_tools_backward_compat(self):
        """Async prepare_data without tools → no tool keys."""
        engine = _make_engine()
        chat = ModelChat()
        chat.add_user_message("Hello")

        json_bytes, headers, data = await engine.prepare_data(chat)

        assert "tools" not in data
        assert "toolConfig" not in data


# ─── process_generate() ────────────────────────────────────────────────────


class TestProcessGenerate:
    """process_generate() handling of functionCall parts."""

    def test_native_id_in_function_call(self):
        """functionCall with native id → tool_call uses native ID."""
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "id": "native_id_123",
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "native_id_123"
        assert result.tool_calls[0].function.name == "get_weather"

    def test_no_id_uses_synthetic_id(self):
        """functionCall without id → tool_call uses synthetic ID."""
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        # Synthetic ID format: call_{n}_{timestamp}
        assert result.tool_calls[0].id.startswith("call_0_")

    def test_missing_args_defaults_to_empty_object(self):
        """functionCall without args → arguments = "{}"."""
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            # No "args" key
                        }
                    }]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        assert result.tool_calls is not None
        assert result.tool_calls[0].function.arguments == "{}"

    def test_missing_name_skips_part(self):
        """functionCall without name → part skipped, no crash."""
        response = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me check..."},
                        {
                            "functionCall": {
                                # No "name" key
                                "args": {"city": "London"},
                            }
                        },
                    ]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        engine = _make_engine()
        result = engine.process_generate(response)

        # Text should be present, tool_calls should be None (skipped the malformed part)
        assert result.content == "Let me check..."
        assert result.tool_calls is None


# ─── prepare_stream_response() ─────────────────────────────────────────────


class TestPrepareStreamResponse:
    """prepare_stream_response() handling of streaming chunks."""

    def test_text_part(self):
        """Chunk with text part → delta.content = text."""
        chunk = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        })
        chunk = f"data: {chunk}"

        engine = _make_engine()
        result = engine.prepare_stream_response(chunk)

        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].delta.tool_calls is None

    def test_function_call_part(self):
        """Chunk with functionCall part → delta.content = "", tool_calls populated."""
        chunk = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "id": "stream_call_1",
                            "name": "get_weather",
                            "args": {"city": "London"},
                        }
                    }],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        })
        chunk = f"data: {chunk}"

        engine = _make_engine()
        result = engine.prepare_stream_response(chunk)

        assert result.choices[0].delta.content == ""
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0].function.name == "get_weather"

    def test_mixed_parts_text_and_function_call(self):
        """Chunk with text + functionCall → delta.content = text, tool_calls populated."""
        chunk = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Checking..."},
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "London"},
                            }
                        },
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        })
        chunk = f"data: {chunk}"

        engine = _make_engine()
        result = engine.prepare_stream_response(chunk)

        assert result.choices[0].delta.content == "Checking..."
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1

    def test_function_call_missing_name_skipped(self):
        """Streaming functionCall without name → skipped, no crash."""
        chunk = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            # No "name" key
                            "args": {"city": "London"},
                        }
                    }],
                    "role": "model",
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        })
        chunk = f"data: {chunk}"

        engine = _make_engine()
        result = engine.prepare_stream_response(chunk)

        # Should not crash, content should be empty, no tool_calls
        assert result.choices[0].delta.content == ""
        assert result.choices[0].delta.tool_calls is None
