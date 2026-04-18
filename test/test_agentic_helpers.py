"""Tests for agentic helper functions (magic_llm.util.agentic).

Covers:
- Slice 19: _coerce_tool_output
- Slice 20: _parse_tool_arguments
- Slice 21: _safe_preview
- Slice 22: _build_tool_registry
- Slice A: _format_tool_feedback (TDD: agentic-tooling-flow-tests)
- Slice B: _create_separator_chunk (TDD: agentic-tooling-flow-tests)
"""

import json
from unittest.mock import MagicMock

import pytest

from magic_llm.util.agentic import (
    _coerce_tool_output,
    _parse_tool_arguments,
    _safe_preview,
    _build_tool_registry,
    _format_tool_feedback,
    _create_separator_chunk,
)


# ─── Slice 19: _coerce_tool_output ─────────────────────────────────────────

class TestCoerceToolOutput:
    """_coerce_tool_output always returns a string representation."""

    def test_dict_to_json_string(self):
        result = _coerce_tool_output({"key": "value", "num": 42})
        parsed = json.loads(result)
        assert parsed == {"key": "value", "num": 42}

    def test_list_to_json_string(self):
        result = _coerce_tool_output([1, 2, 3])
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_string_passthrough(self):
        result = _coerce_tool_output("hello")
        assert result == "hello"

    def test_int_to_string(self):
        result = _coerce_tool_output(42)
        assert result == "42"

    def test_none_to_string(self):
        result = _coerce_tool_output(None)
        assert result == "None"

    def test_float_to_string(self):
        result = _coerce_tool_output(3.14)
        assert result == "3.14"

    def test_bool_to_string(self):
        assert _coerce_tool_output(True) == "True"
        assert _coerce_tool_output(False) == "False"

    def test_circular_ref_fallback(self):
        """Circular references should fall back to str()."""
        lst = []
        lst.append(lst)  # circular
        result = _coerce_tool_output(lst)
        assert isinstance(result, str)
        assert len(result) > 0


# ─── Slice 20: _parse_tool_arguments ───────────────────────────────────────

class TestParseToolArguments:
    """_parse_tool_arguments parses JSON argument strings."""

    def test_valid_json_string(self):
        parsed, echo = _parse_tool_arguments('{"name": "Alice", "age": 30}')
        assert parsed == {"name": "Alice", "age": 30}
        assert echo == '{"name": "Alice", "age": 30}'

    def test_invalid_json_returns_raw(self):
        parsed, echo = _parse_tool_arguments("not json at all")
        assert parsed == "not json at all"
        assert echo == "not json at all"

    def test_none_returns_empty(self):
        parsed, echo = _parse_tool_arguments(None)
        assert parsed is None
        assert echo == ""

    def test_json_array(self):
        parsed, echo = _parse_tool_arguments('[1, 2, 3]')
        assert parsed == [1, 2, 3]

    def test_json_nested_object(self):
        parsed, echo = _parse_tool_arguments('{"outer": {"inner": "value"}}')
        assert parsed == {"outer": {"inner": "value"}}


# ─── Slice 21: _safe_preview ───────────────────────────────────────────────

class TestSafePreview:
    """_safe_preview returns a truncated, printable preview."""

    def test_short_string_unchanged(self):
        result = _safe_preview("hello", max_len=800)
        assert result == "hello"

    def test_long_string_truncated(self):
        long_str = "a" * 1000
        result = _safe_preview(long_str, max_len=100)
        assert len(result) <= 100 + len("... [truncated 900 chars]")
        assert result.startswith("a" * 100)
        assert "... [truncated" in result

    def test_truncated_char_count_correct(self):
        long_str = "x" * 200
        result = _safe_preview(long_str, max_len=100)
        assert "... [truncated 100 chars]" in result

    def test_dict_to_json(self):
        result = _safe_preview({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_non_serializable_fallback_to_str(self):
        obj = object()
        result = _safe_preview(obj)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_default_max_len(self):
        long_str = "b" * 1000
        result = _safe_preview(long_str)  # default max_len=800
        assert len(result) < 1000
        assert "... [truncated" in result


# ─── Slice 22: _build_tool_registry ────────────────────────────────────────

class TestBuildToolRegistry:
    """_build_tool_registry builds a name->callable dict from tools."""

    def test_callable_registered_by_name(self):
        def my_tool(x):
            return x

        registry = _build_tool_registry([my_tool], None)
        assert "my_tool" in registry
        assert registry["my_tool"] is my_tool

    def test_empty_inputs_returns_empty_dict(self):
        registry = _build_tool_registry(None, None)
        assert registry == {}

    def test_tool_functions_merged(self):
        def tool_a():
            pass

        def tool_b():
            pass

        registry = _build_tool_registry([tool_a], {"tool_b": tool_b})
        assert "tool_a" in registry
        assert "tool_b" in registry

    def test_tool_functions_override(self):
        def original():
            return "original"

        def override():
            return "override"

        registry = _build_tool_registry([original], {"original": override})
        assert registry["original"] is override

    def test_dict_tool_with_function_map(self):
        def my_func():
            pass

        tool_spec = {
            "type": "function",
            "function": {"name": "my_func"},
        }
        registry = _build_tool_registry([tool_spec], {"my_func": my_func})
        assert "my_func" in registry
        assert registry["my_func"] is my_func

    def test_dict_tool_legacy_format(self):
        def legacy_func():
            pass

        tool_spec = {"name": "legacy_func"}
        registry = _build_tool_registry([tool_spec], {"legacy_func": legacy_func})
        assert "legacy_func" in registry

    def test_dict_tool_without_function_map_ignored(self):
        tool_spec = {
            "type": "function",
            "function": {"name": "missing_func"},
        }
        registry = _build_tool_registry([tool_spec], None)
        assert "missing_func" not in registry

    def test_non_callable_ignored(self):
        registry = _build_tool_registry(["not_callable", 42, None], None)
        assert registry == {}

    def test_callable_without_name_ignored(self):
        # Lambda has __name__ = '<lambda>' which is truthy, so it IS registered
        fn = lambda x: x  # noqa: E731
        registry = _build_tool_registry([fn], None)
        assert "<lambda>" in registry


# ─── Slice A: _format_tool_feedback ────────────────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 1, Pure Unit

class TestFormatToolFeedback:
    """_format_tool_feedback produces the current plain-text tool result format.

    NOTE: This is the CURRENT behavior. Tool results are formatted as plain text
    and later injected as role:"user" messages. The SDD refactor will change this
    to proper role:"tool" messages with tool_call_id.
    """

    def test_normal_args_and_output(self):
        result = _format_tool_feedback(
            name="get_weather",
            input_str='{"city": "London"}',
            output_str='{"temp": 18, "unit": "C"}',
        )
        assert result == (
            "tool: get_weather\n"
            "tool input: {\"city\": \"London\"}\n"
            "tool output: {\"temp\": 18, \"unit\": \"C\"}"
        )

    def test_empty_input(self):
        result = _format_tool_feedback(
            name="list_items",
            input_str="",
            output_str="[]",
        )
        assert "tool: list_items" in result
        assert "tool input: " in result
        assert "tool output: []" in result

    def test_empty_output(self):
        result = _format_tool_feedback(
            name="void_fn",
            input_str="{}",
            output_str="",
        )
        assert "tool: void_fn" in result
        assert "tool output: " in result

    def test_multiline_output(self):
        result = _format_tool_feedback(
            name="run_script",
            input_str="{}",
            output_str="line1\nline2\nline3",
        )
        assert "tool: run_script" in result
        assert "line1\nline2\nline3" in result

    def test_json_error_output(self):
        """Error feedback from unknown tool or exception uses JSON error dict."""
        result = _format_tool_feedback(
            name="unknown_tool",
            input_str="{}",
            output_str='{"error": "Unknown tool: unknown_tool"}',
        )
        assert "tool: unknown_tool" in result
        assert '"error"' in result


# ─── Slice B: _create_separator_chunk ──────────────────────────────────────
# TDD: agentic-tooling-flow-tests — Phase 1, Pure Unit

class TestCreateSeparatorChunk:
    """_create_separator_chunk creates a valid ChatCompletionModel with separator content."""

    def test_normal_construction(self):
        from magic_llm.model.ModelChatStream import ChatCompletionModel
        chunk = _create_separator_chunk(
            separator="\n\n",
            model="test-model",
            chunk_id="sep-1",
        )
        assert isinstance(chunk, ChatCompletionModel)
        assert chunk.id == "sep-1"
        assert chunk.model == "test-model"
        assert len(chunk.choices) == 1
        assert chunk.choices[0].delta.content == "\n\n"

    def test_empty_model_fallback(self):
        chunk = _create_separator_chunk(
            separator="---",
            model="",
            chunk_id="sep-2",
        )
        assert chunk.model == ""
        assert chunk.id == "sep-2"
        assert chunk.choices[0].delta.content == "---"

    def test_custom_separator(self):
        chunk = _create_separator_chunk(
            separator="<SEPARATOR>",
            model="gpt-4",
            chunk_id="custom-id",
        )
        assert chunk.choices[0].delta.content == "<SEPARATOR>"
