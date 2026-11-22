"""Unit tests for map_to_gemini() function in tools_mapping.py.

Tests cover:
- Tools convert to functionDeclarations format with parametersJsonSchema + additionalProperties: false
- tool_choice mapping: auto→AUTO, required→ANY, none→NONE, {"name": "foo"}→ANY+allowedFunctionNames
- Empty tools with tool_choice → gemini_tools=None, gemini_tool_config set
- Schema with local $ref → inlined, no $defs in output
- tool_choice=None → gemini_tool_config=None
"""

from typing import Optional

import pytest

from pydantic import BaseModel

from magic_llm.util.tools_mapping import map_to_gemini


# ─── Helpers ────────────────────────────────────────────────────────────────


def _get_weather(city: str) -> str:
    """Get weather for a city."""
    return "sunny"


# ─── Basic tool conversion ─────────────────────────────────────────────────


class TestMapToGeminiBasic:
    """Basic tool conversion to functionDeclarations format."""

    def test_callable_tool(self):
        """Callable tool → functionDeclarations with parametersJsonSchema."""
        gemini_tools, gemini_tool_config = map_to_gemini([_get_weather], tool_choice=None)

        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        tool = gemini_tools[0]
        assert tool["name"] == "_get_weather"
        assert tool["description"] == "Get weather for a city."
        assert "parametersJsonSchema" in tool

        schema = tool["parametersJsonSchema"]
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert "city" in schema["required"]
        assert schema["additionalProperties"] is False

        assert gemini_tool_config is None

    def test_dict_tool_spec(self):
        """Dict spec → correct format with additionalProperties: false."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }

        gemini_tools, _ = map_to_gemini([tool_spec], tool_choice=None)

        assert gemini_tools is not None
        assert len(gemini_tools) == 1
        tool = gemini_tools[0]
        assert tool["name"] == "search"
        assert tool["parametersJsonSchema"]["additionalProperties"] is False

    def test_empty_tools(self):
        """Empty tools → gemini_tools=None."""
        gemini_tools, gemini_tool_config = map_to_gemini([], tool_choice=None)
        assert gemini_tools is None
        assert gemini_tool_config is None


# ─── tool_choice mapping ───────────────────────────────────────────────────


class TestMapToGeminiToolChoice:
    """tool_choice mapping to functionCallingConfig mode."""

    def test_tool_choice_auto(self):
        """tool_choice="auto" → {"functionCallingConfig": {"mode": "AUTO"}}."""
        _, gemini_tool_config = map_to_gemini([_get_weather], tool_choice="auto")
        assert gemini_tool_config == {"functionCallingConfig": {"mode": "AUTO"}}

    def test_tool_choice_required(self):
        """tool_choice="required" → {"functionCallingConfig": {"mode": "ANY"}}."""
        _, gemini_tool_config = map_to_gemini([_get_weather], tool_choice="required")
        assert gemini_tool_config == {"functionCallingConfig": {"mode": "ANY"}}

    def test_tool_choice_none_string(self):
        """tool_choice="none" → {"functionCallingConfig": {"mode": "NONE"}}."""
        _, gemini_tool_config = map_to_gemini([_get_weather], tool_choice="none")
        assert gemini_tool_config == {"functionCallingConfig": {"mode": "NONE"}}

    def test_tool_choice_named_function(self):
        """tool_choice={"name": "get_weather"} → mode ANY + allowedFunctionNames."""
        _, gemini_tool_config = map_to_gemini([_get_weather], tool_choice={"name": "get_weather"})
        assert gemini_tool_config == {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": ["get_weather"],
            },
        }

    def test_tool_choice_none_value(self):
        """tool_choice=None → gemini_tool_config=None."""
        _, gemini_tool_config = map_to_gemini([_get_weather], tool_choice=None)
        assert gemini_tool_config is None

    def test_empty_tools_with_tool_choice(self):
        """Empty tools with tool_choice → gemini_tools=None, gemini_tool_config set."""
        gemini_tools, gemini_tool_config = map_to_gemini([], tool_choice="auto")
        assert gemini_tools is None
        assert gemini_tool_config == {"functionCallingConfig": {"mode": "AUTO"}}


# ─── $ref inlining ─────────────────────────────────────────────────────────


class TestMapToGeminiRefInlining:
    """Schema with local $ref → inlined, no $defs in output."""

    def test_pydantic_model_with_nested_types(self):
        """Pydantic model with nested model → inlined schema, no $defs."""

        class Address(BaseModel):
            """Address details."""
            street: str
            city: str

        class Person(BaseModel):
            """A person with an address."""
            name: str
            address: Address

        gemini_tools, _ = map_to_gemini([Person], tool_choice=None)

        assert gemini_tools is not None
        tool = gemini_tools[0]
        schema = tool["parametersJsonSchema"]

        # No $defs or definitions at top level
        assert "$defs" not in schema
        assert "definitions" not in schema

        # Address should be inlined
        assert "address" in schema["properties"]
        address_schema = schema["properties"]["address"]
        assert address_schema.get("type") == "object"
        assert "street" in address_schema.get("properties", {})
        assert "city" in address_schema.get("properties", {})

        # additionalProperties: false
        assert schema["additionalProperties"] is False
