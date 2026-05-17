import json
import pytest

from magic_llm.exception.ChatException import ChatException
from magic_llm.engine.tooling import (
    _extract_param_docs_from_docstring,
    _schema_from_callable,
    normalize_openai_tools,
    normalize_openai_tool_choice,
    map_to_openai,
    map_to_anthropic,
    coerce_tool_choice_to_string,
)
from magic_llm.util import tools_mapping as compat_tools_mapping


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


def test_normalize_openai_tools_mixed_forms():
    tools = [
        FUNCTION_DEF_NEW,
        FUNCTION_DEF_LEGACY,
        None,
    ]
    out = normalize_openai_tools(tools)

    # Only one canonical entry (duplicates by name allowed but we assert correctness)
    assert isinstance(out, list) and len(out) == 2
    for t in out:
        assert set(t.keys()) == {"name", "description", "parameters"}
        assert t["name"] == "get_weather"
        # parameters is a schema-like dict
        assert isinstance(t["parameters"], dict)


def test_normalize_openai_tools_invalid_dict_fails_descriptively():
    with pytest.raises(ChatException) as exc_info:
        normalize_openai_tools([{"description": "missing name"}])

    assert exc_info.value.error_code == "TOOL_NORMALIZATION_ERROR"
    assert "Unsupported tool definition" in str(exc_info.value)
    assert "missing tool name" in str(exc_info.value)


def test_old_util_imports_delegate_to_engine_tooling():
    """Compatibility wrapper remains, but engine/core owns the implementation."""

    assert compat_tools_mapping.normalize_openai_tools is normalize_openai_tools
    assert compat_tools_mapping.map_to_openai is map_to_openai


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


def test_callable_name_docstring_type_hints_and_param_docs_from_engine_tooling():
    def get_weather(city: str, days: int = 1) -> str:
        """Get current weather.

        Args:
            city: City to inspect.
            days: Forecast window in days.
        """
        return ""

    out = normalize_openai_tools([get_weather])

    tool = out[0]
    assert tool["name"] == "get_weather"
    assert tool["description"].startswith("Get current weather.")
    params = tool["parameters"]
    assert params["properties"]["city"]["type"] == "string"
    assert params["properties"]["city"]["description"] == "City to inspect."
    assert params["properties"]["days"]["type"] == "integer"
    assert params["properties"]["days"]["description"] == "Forecast window in days."
    assert params["required"] == ["city"]


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


def test_pydantic_field_descriptions_and_nested_refs_are_preserved_by_engine_tooling():
    try:
        from pydantic import BaseModel, Field
    except Exception:
        pytest.skip("pydantic not available")

    class Location(BaseModel):
        """Location payload."""
        city: str = Field(description="City name")

    class ForecastRequest(BaseModel):
        """Forecast request."""
        location: Location
        days: int = Field(description="Number of forecast days")

    out = normalize_openai_tools([ForecastRequest])
    params = out[0]["parameters"]

    assert out[0]["name"] == "ForecastRequest"
    assert "Forecast request" in out[0]["description"]
    assert "$defs" not in params
    assert "$ref" not in json.dumps(params)
    assert params["properties"]["days"]["description"] == "Number of forecast days"
    location = params["properties"]["location"]
    assert location["properties"]["city"]["description"] == "City name"


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


# =============================================================================
# Phase 3A: _extract_param_docs_from_docstring() tests
# =============================================================================


class TestExtractParamDocsFromDocstring:
    """Direct unit tests for _extract_param_docs_from_docstring()."""

    def test_rest_params(self):
        """reST/Sphinx :param style with multiple params."""
        doc = ":param location: The city name.\n:param unit: Celsius or Fahrenheit."
        result = _extract_param_docs_from_docstring(doc)
        assert result == {
            "location": "The city name.",
            "unit": "Celsius or Fahrenheit.",
        }

    def test_google_style(self):
        """Google-style Args section with types."""
        doc = (
            "Args:\n"
            "    location (str): The city name.\n"
            "    unit (str): Celsius or Fahrenheit."
        )
        result = _extract_param_docs_from_docstring(doc)
        assert result == {
            "location": "The city name.",
            "unit": "Celsius or Fahrenheit.",
        }

    def test_multiline_reST(self):
        """reST continuation lines join into single description."""
        doc = (
            ":param name: A long description that\n"
            "    continues on the next line."
        )
        result = _extract_param_docs_from_docstring(doc)
        assert result == {"name": "A long description that continues on the next line."}

    def test_multiline_google(self):
        """Google-style continuation lines join into single description."""
        doc = (
            "Args:\n"
            "    name (str): A long description\n"
            "        that continues on the next\n"
            "        line."
        )
        result = _extract_param_docs_from_docstring(doc)
        assert result == {"name": "A long description that continues on the next line."}

    def test_empty(self):
        """Empty docstring returns empty dict."""
        assert _extract_param_docs_from_docstring("") == {}

    def test_no_params(self):
        """Docstring with no parameter section returns empty dict."""
        assert _extract_param_docs_from_docstring("Just a description.") == {}

    def test_mixed_formats(self):
        """Both reST and Google-style: reST pass (pass1) takes priority."""
        doc = (
            ":param location: The city name.\n"
            ":param unit: Celsius or Fahrenheit.\n"
            "\n"
            "Args:\n"
            "    location (str): OVERRIDE.\n"
            "    unit (str): OVERRIDE."
        )
        result = _extract_param_docs_from_docstring(doc)
        # reST values take priority; Google entries for already-known names are skipped
        assert result == {
            "location": "The city name.",
            "unit": "Celsius or Fahrenheit.",
        }

    def test_unicode(self):
        """Unicode characters in param names and descriptions."""
        doc = (
            ":param città: Nome della città.\n"
            ":param température: Gradi Celsius."
        )
        result = _extract_param_docs_from_docstring(doc)
        assert result == {
            "città": "Nome della città.",
            "température": "Gradi Celsius.",
        }

    def test_google_no_type(self):
        """Google-style without type annotation: 'name: desc' works."""
        doc = (
            "Args:\n"
            "    location: The city name.\n"
            "    unit: Celsius or Fahrenheit."
        )
        result = _extract_param_docs_from_docstring(doc)
        assert result == {
            "location": "The city name.",
            "unit": "Celsius or Fahrenheit.",
        }


# =============================================================================
# Phase 3A: _schema_from_callable() tests
# =============================================================================


class TestSchemaFromCallable:
    """Direct unit tests for _schema_from_callable()."""

    def test_with_docstring(self):
        """Function with docstring: description extracted correctly."""
        def get_weather(location: str, unit: str = "C"):
            """Get current temperature for a given location."""
            pass

        name, desc, params = _schema_from_callable(get_weather)
        assert name == "get_weather"
        assert desc == "Get current temperature for a given location."
        assert "location" in params["properties"]
        assert "unit" in params["properties"]
        assert params["type"] == "object"

    def test_no_docstring(self):
        """Function with no docstring yields empty description."""
        def get_weather(location: str):
            pass

        name, desc, _ = _schema_from_callable(get_weather)
        assert desc == ""

    def test_no_type_hints(self):
        """Function without annotations: params still extracted, default to string type."""
        def get_weather(location, unit="C"):
            """Get the weather."""
            pass

        name, desc, params = _schema_from_callable(get_weather)
        assert desc == "Get the weather."
        assert "location" in params["properties"]
        assert "unit" in params["properties"]
        # No annotation → _json_schema_for_annotation returns {} → fallback to {"type": "string"}
        assert params["properties"]["location"].get("type") == "string"

    def test_optional_type(self):
        """Optional[str] yields string type and not required."""
        from typing import Optional

        def get_weather(location: Optional[str] = None):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        # Optional[str] → Union[str, None] → non-None inner type → string
        assert params["properties"]["location"]["type"] == "string"
        assert "location" not in params["required"]

    def test_union_multiple_types(self):
        """Union[str, int] with multiple non-None types falls back to default."""
        from typing import Union

        def get_weather(location: Union[str, int]):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        # Union[str, int] → multi non-None → _json_schema_for_annotation returns {} → {"type": "string"}
        assert "location" in params["properties"]

    def test_list_type(self):
        """List[int] yields array schema with integer items."""
        from typing import List

        def get_weather(cities: List[int]):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        assert params["properties"]["cities"]["type"] == "array"
        assert params["properties"]["cities"]["items"]["type"] == "integer"

    def test_dict_type(self):
        """Dict[str, Any] yields object type."""
        from typing import Dict, Any

        def get_weather(data: Dict[str, Any]):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        assert params["properties"]["data"]["type"] == "object"

    def test_literal_type(self):
        """Literal["a", "b"] yields enum."""
        from typing import Literal

        def get_weather(unit: Literal["C", "F"] = "C"):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        assert params["properties"]["unit"]["enum"] == ["C", "F"]

    def test_kwargs_and_args(self):
        """*args and **kwargs appear as regular properties."""
        def get_weather(location: str, *args: str, **kwargs: int):
            """Get weather."""
            pass

        _, _, params = _schema_from_callable(get_weather)
        assert "location" in params["properties"]
        assert "args" in params["properties"]
        assert "kwargs" in params["properties"]
        assert params["properties"]["args"]["type"] == "string"
        assert params["properties"]["kwargs"]["type"] == "integer"

    def test_self_filtered(self):
        """self parameter excluded from schema properties."""
        class WeatherService:
            def get_weather(self, location: str):
                """Get weather."""
                pass

        _, _, params = _schema_from_callable(WeatherService.get_weather)
        assert "self" not in params["properties"]
        assert "location" in params["properties"]

    def test_cls_filtered(self):
        """cls parameter excluded from schema properties."""
        class WeatherService:
            @classmethod
            def get_weather(cls, location: str):
                """Get weather."""
                pass

        _, _, params = _schema_from_callable(WeatherService.get_weather)
        assert "cls" not in params["properties"]
        assert "location" in params["properties"]

    def test_async_function(self):
        """Async function handled correctly."""
        async def get_weather(location: str):
            """Get weather for a location."""
            pass

        name, desc, params = _schema_from_callable(get_weather)
        assert name == "get_weather"
        assert desc == "Get weather for a location."
        assert "location" in params["properties"]

    def test_param_description_from_docstring(self):
        """Parameter descriptions from docstring attached to properties."""
        def get_weather(location: str, unit: str = "C"):
            """Get current temperature.

            Args:
                location: The city name.
                unit: Celsius or Fahrenheit.
            """
            pass

        _, _, params = _schema_from_callable(get_weather)
        assert params["properties"]["location"].get("description") == "The city name."
        assert params["properties"]["unit"].get("description") == "Celsius or Fahrenheit."


# =============================================================================
# Phase 3A: Anthropic format description assertions
# =============================================================================


def test_map_to_anthropic_tools_description_field():
    """Anthropic format includes correct description for dict-based tools."""
    tools, choice = map_to_anthropic([FUNCTION_DEF_LEGACY], {"name": "get_weather"})
    assert "description" in tools[0], "Anthropic format MUST include 'description' field"
    assert tools[0]["description"] == "Get current temperature for a given location."


def test_map_to_anthropic_from_callable_description():
    """Callable docstring propagates to Anthropic description field."""
    def get_weather(location: str):
        """Get current temperature for a given location."""
        return ""

    tools, choice = map_to_anthropic([get_weather], {"name": "get_weather"})
    assert "description" in tools[0], "Anthropic format MUST include 'description' field"
    assert tools[0]["description"] == "Get current temperature for a given location."
