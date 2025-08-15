import json
import pytest

from magic_llm.util.tools_mapping import (
    normalize_openai_tools,
    normalize_openai_tool_choice,
    map_to_openai,
    map_to_anthropic,
    coerce_tool_choice_to_string,
)


FUNCTION_DEF_NEW = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"],
        },
    },
}

FUNCTION_DEF_LEGACY = {
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"],
    },
}


def test_normalize_openai_tools_mixed_forms_and_invalids():
    tools = [
        FUNCTION_DEF_NEW,
        FUNCTION_DEF_LEGACY,
        None,
        {"description": "missing name"},
    ]
    out = normalize_openai_tools(tools)

    # Only one canonical entry (duplicates by name allowed but we assert correctness)
    assert isinstance(out, list) and len(out) == 2
    for t in out:
        assert set(t.keys()) == {"name", "description", "parameters"}
        assert t["name"] == "get_weather"
        # parameters is a schema-like dict
        assert isinstance(t["parameters"], dict)


def test_normalize_openai_tool_choice_variants():
    # Strings
    assert normalize_openai_tool_choice("auto") == "auto"
    assert normalize_openai_tool_choice("required") == "required"
    assert normalize_openai_tool_choice("none") == "none"

    # Unknown string: passthrough
    assert normalize_openai_tool_choice("unknown-mode") == "unknown-mode"

    # Wrapped and legacy dict
    assert normalize_openai_tool_choice({"type": "function", "function": {"name": "x"}}) == {"name": "x"}
    assert normalize_openai_tool_choice({"name": "y"}) == {"name": "y"}

    # Invalid dict / None
    assert normalize_openai_tool_choice({}) is None
    assert normalize_openai_tool_choice(None) is None


def test_map_to_openai_from_legacy_and_choice_dict():
    tools = [FUNCTION_DEF_LEGACY]
    tool_choice = {"name": "get_weather"}

    openai_tools, openai_choice = map_to_openai(tools, tool_choice)

    assert isinstance(openai_tools, list) and len(openai_tools) == 1
    t0 = openai_tools[0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "get_weather"
    assert "parameters" in t0["function"]

    assert openai_choice == {"type": "function", "function": {"name": "get_weather"}}


def test_map_to_openai_choice_strings_passthrough():
    for s in ("auto", "none", "required"):
        _, c = map_to_openai([FUNCTION_DEF_LEGACY], s)
        assert c == s


def test_map_to_anthropic_tools_and_choice_mapping():
    tools = [FUNCTION_DEF_LEGACY]

    # named function
    t, c = map_to_anthropic(tools, {"name": "get_weather"})
    assert isinstance(t, list) and t[0]["name"] == "get_weather"
    assert "input_schema" in t[0]
    assert c == {"type": "tool", "name": "get_weather"}

    # auto -> {type:auto}
    t, c = map_to_anthropic(tools, "auto")
    assert c == {"type": "auto"}

    # required -> {type:any}
    t, c = map_to_anthropic(tools, "required")
    assert c == {"type": "any"}

    # none -> None
    t, c = map_to_anthropic(tools, "none")
    assert c is None


def test_coerce_tool_choice_to_string_behaviors():
    # dict -> downgraded to default
    assert coerce_tool_choice_to_string({"name": "x"}, default="auto") == "auto"

    # string preserved
    assert coerce_tool_choice_to_string("required") == "required"

    # None -> None
    assert coerce_tool_choice_to_string(None) is None
