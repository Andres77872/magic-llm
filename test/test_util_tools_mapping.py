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


def test_normalize_openai_tools_accepts_callable_and_schema_alias():
    # Python callable
    def get_weather(location: str, unit: str = "C"):
        """Get current temperature for a given location."""
        return ""

    # Dict with schema alias instead of parameters
    tool_with_schema = {
        "name": "get_weather_alt",
        "description": "Get current temperature for a given location.",
        "schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }

    out = normalize_openai_tools([get_weather, tool_with_schema])
    # Expect two canonical entries
    assert isinstance(out, list) and len(out) == 2

    names = {t["name"] for t in out}
    assert {"get_weather", "get_weather_alt"}.issubset(names)

    # Validate callable schema inference
    callable_entry = next(t for t in out if t["name"] == "get_weather")
    assert callable_entry["parameters"]["type"] == "object"
    assert "location" in callable_entry["parameters"]["properties"]


def test_normalize_openai_tools_accepts_pydantic_model():
    try:
        from pydantic import BaseModel
    except Exception:
        pytest.skip("pydantic not available")

    class GetWeather(BaseModel):
        """Get current temperature for a given location."""
        location: str

    out = normalize_openai_tools([GetWeather])
    assert isinstance(out, list) and len(out) == 1
    t0 = out[0]
    # name uses class name by default
    assert t0["name"] == "GetWeather"
    assert t0["parameters"]["type"] == "object"


def test_map_to_openai_from_callable():
    def get_weather(location: str):
        """Get current temperature for a given location."""
        return ""

    tools, choice = map_to_openai([get_weather], {"name": "get_weather"})
    assert isinstance(tools, list) and tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert "parameters" in tools[0]["function"]
    assert choice == {"type": "function", "function": {"name": "get_weather"}}


def test_map_to_anthropic_from_callable():
    def get_weather(location: str):
        """Get current temperature for a given location."""
        return ""

    tools, choice = map_to_anthropic([get_weather], {"name": "get_weather"})
    assert isinstance(tools, list)
    assert tools[0]["name"] == "get_weather"
    assert "input_schema" in tools[0]
    assert choice == {"type": "tool", "name": "get_weather"}


def test_normalize_preserves_strict_flag():
    # Wrapper style with strict at wrapper level
    tool_wrapper = {
        "type": "function",
        "strict": True,
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    out = normalize_openai_tools([tool_wrapper])
    assert isinstance(out, list) and out[0]["name"] == "get_weather"
    assert out[0].get("strict") is True

    # Strict inside the function block also preserved
    tool_inside = {
        "type": "function",
        "function": {
            "name": "get_weather2",
            "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    out2 = normalize_openai_tools([tool_inside])
    assert out2[0]["name"] == "get_weather2" and out2[0].get("strict") is True


def test_callable_param_with_nested_pydantic_schema_includes_inner_properties():
    try:
        from pydantic import BaseModel
    except Exception:
        pytest.skip("pydantic not available")

    class InnerModel(BaseModel):
        data: str

    class SomeModel(BaseModel):
        inner_model: InnerModel

    def use_nested(some_model: SomeModel):
        """Tool with nested pydantic param."""
        return ""

    out = normalize_openai_tools([use_nested])
    print(out)
    assert isinstance(out, list) and len(out) == 1
    entry = out[0]
    assert entry["name"] == "use_nested"
    params = entry["parameters"]
    assert params.get("type") == "object"
    # the outer callable param should be required
    assert "some_model" in (params.get("required") or [])
    assert "some_model" in params.get("properties", {})
    sm_schema = params["properties"]["some_model"]
    # Expect SomeModel object schema
    assert sm_schema.get("type") == "object"
    # Must not contain local $defs/definitions or $ref anywhere
    def _no_ref_defs(node):
        if isinstance(node, dict):
            assert "$defs" not in node and "definitions" not in node and "$ref" not in node
            for v in node.values():
                _no_ref_defs(v)
        elif isinstance(node, list):
            for v in node:
                _no_ref_defs(v)

    _no_ref_defs(sm_schema)

    # inner_model property exists and is fully inlined
    assert "properties" in sm_schema
    assert "inner_model" in sm_schema["properties"]
    # SomeModel requires inner_model
    assert "inner_model" in (sm_schema.get("required") or [])
    inner_entry = sm_schema["properties"]["inner_model"]
    assert inner_entry.get("type") == "object"
    assert "data" in (inner_entry.get("properties") or {})
    # InnerModel requires data
    assert "data" in (inner_entry.get("required") or [])


def test_map_to_openai_preserves_strict_flag():
    tool = {
        "type": "function",
        "strict": True,
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    tools, choice = map_to_openai([tool], {"name": "get_weather"})
    assert isinstance(tools, list)
    f = tools[0]["function"]
    assert f.get("strict") is True
