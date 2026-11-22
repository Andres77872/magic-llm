"""Tests for builtin agent-loop todo tools."""

import pytest

from magic_llm.agent import config
from magic_llm.agent.builtin_tools import (
    TODOREAD_TOOL_SCHEMA,
    TODOWRITE_TOOL_SCHEMA,
    TodoState,
    builtin_todo_tool_schemas,
    create_builtin_todo_tools,
)


@pytest.fixture(autouse=True)
def restore_builtin_todo_flag():
    original = config.ENABLE_BUILTIN_TODO_TOOLS
    yield
    config.ENABLE_BUILTIN_TODO_TOOLS = original


def _todo(todo_id=1, content="Do work", status="in_progress", priority="high"):
    return {"id": todo_id, "content": content, "status": status, "priority": priority}


class TestTodoState:
    def test_initial_snapshot_is_empty(self):
        assert TodoState().snapshot() == {"ok": True, "todos": []}

    def test_successful_replace_preserves_order_and_ids(self):
        state = TodoState()
        todos = [
            _todo(10, " First ", "in_progress", "high"),
            _todo(11, "Second", "pending", "medium"),
            _todo(12, "Third", "cancelled", "low"),
        ]

        result = state.replace(todos)

        assert result["ok"] is True
        assert [item["id"] for item in result["todos"]] == [10, 11, 12]
        assert [item["content"] for item in state.snapshot()["todos"]] == [
            "First",
            "Second",
            "Third",
        ]

    @pytest.mark.parametrize(
        "todos, match",
        [
            ({}, "array"),
            ([{"content": "x", "status": "pending", "priority": "low"}], "id"),
            ([_todo(True)], "integer"),
            ([_todo(1), _todo(1, "Other", "pending")], "duplicate"),
            ([_todo(1, "   ")], "non-empty"),
            ([_todo(1, status="blocked")], "status"),
            ([_todo(1, priority="urgent")], "priority"),
            ([_todo(1, status="in_progress"), _todo(2, "B", "in_progress")], "at most one"),
        ],
    )
    def test_invalid_replace_rejected_without_mutating_prior_state(self, todos, match):
        state = TodoState()
        original = [_todo(99, "Original", "completed", "medium")]
        state.replace(original)

        with pytest.raises(ValueError, match=match):
            state.replace(todos)

        assert state.snapshot()["todos"] == original

    def test_zero_in_progress_allowed_for_terminal_list(self):
        state = TodoState()
        terminal = [_todo(1, "Done", "completed"), _todo(2, "Nope", "cancelled")]
        assert state.replace(terminal)["todos"] == terminal


class TestBuiltinToolSchemasAndCallables:
    def test_schema_names_and_required_fields(self):
        schemas = builtin_todo_tool_schemas()
        assert [schema["function"]["name"] for schema in schemas] == [
            "todowrite",
            "todoread",
        ]

        write_params = TODOWRITE_TOOL_SCHEMA["function"]["parameters"]
        assert write_params["required"] == ["todos"]
        assert write_params["additionalProperties"] is False
        item_schema = write_params["properties"]["todos"]["items"]
        assert item_schema["required"] == ["id", "content", "status", "priority"]
        assert item_schema["properties"]["id"]["type"] == "integer"

        read_params = TODOREAD_TOOL_SCHEMA["function"]["parameters"]
        assert read_params == {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def test_callables_have_exact_names_and_return_ok_todos(self):
        tools = create_builtin_todo_tools(TodoState())
        assert tools["todowrite"].__name__ == "todowrite"
        assert tools["todoread"].__name__ == "todoread"

        todos = [_todo()]
        assert tools["todowrite"](todos=todos) == {"ok": True, "todos": todos}
        assert tools["todoread"]() == {"ok": True, "todos": todos}

    def test_callables_reject_missing_or_unexpected_arguments(self):
        tools = create_builtin_todo_tools(TodoState())
        with pytest.raises(ValueError, match="requires a todos"):
            tools["todowrite"]()
        with pytest.raises(ValueError, match="unexpected"):
            tools["todowrite"](todos=[], extra=True)
        with pytest.raises(ValueError, match="unexpected"):
            tools["todoread"](filter="all")


class TestBuiltinTodoConfig:
    def test_builtin_todo_tools_default_enabled_and_helpers_toggle(self):
        config.ENABLE_BUILTIN_TODO_TOOLS = True
        assert config.is_builtin_todo_tools_enabled() is True

        config.disable_builtin_todo_tools()
        assert config.is_builtin_todo_tools_enabled() is False

        config.enable_builtin_todo_tools()
        assert config.is_builtin_todo_tools_enabled() is True
