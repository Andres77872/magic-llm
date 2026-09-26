"""Run-local builtin agent-loop tools."""

from __future__ import annotations

import copy
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable

TODO_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
TODO_PRIORITIES = {"high", "medium", "low"}
MAX_TODO_ITEMS = 100
MAX_TODO_CONTENT_LENGTH = 2000
TODO_TOOL_NAMES = {"todowrite", "todoread"}


@dataclass(frozen=True)
class TodoItem:
    id: int
    content: str
    status: str
    priority: str


class TodoState:
    """Thread-safe in-memory todo state scoped to one agent run."""

    def __init__(self) -> None:
        self._todos: list[TodoItem] = []
        self._lock = threading.RLock()

    def replace(self, todos: Any) -> dict[str, Any]:
        """Validate and atomically replace the full todo list."""
        validated = _validate_todos(todos)
        with self._lock:
            self._todos = validated
            return {"ok": True, "todos": [asdict(todo) for todo in self._todos]}

    def snapshot(self) -> dict[str, Any]:
        """Return the current todo list without mutating it."""
        with self._lock:
            return {"ok": True, "todos": [asdict(todo) for todo in self._todos]}


def _validate_todos(todos: Any) -> list[TodoItem]:
    if not isinstance(todos, list):
        raise ValueError("todos must be an array")

    if len(todos) > MAX_TODO_ITEMS:
        raise ValueError(f"todos may contain at most {MAX_TODO_ITEMS} items")

    validated: list[TodoItem] = []
    seen_ids: set[int] = set()
    in_progress_count = 0

    for index, item in enumerate(todos):
        if not isinstance(item, dict):
            raise ValueError(f"todo at index {index} must be an object")

        extra = set(item) - {"id", "content", "status", "priority"}
        if extra:
            raise ValueError(f"todo at index {index} has unexpected fields: {sorted(extra)}")
        missing = {"id", "content", "status", "priority"} - set(item)
        if missing:
            raise ValueError(
                f"todo at index {index} is missing required field(s): "
                f"{', '.join(sorted(missing))}"
            )

        todo_id = item["id"]
        if not isinstance(todo_id, int) or isinstance(todo_id, bool):
            raise ValueError(f"todo at index {index} id must be an integer")
        if todo_id in seen_ids:
            raise ValueError(f"duplicate todo id: {todo_id}")
        seen_ids.add(todo_id)

        content = item["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"todo at index {index} content must be a non-empty string")

        if len(content) > MAX_TODO_CONTENT_LENGTH:
            raise ValueError(f"todo content exceeds {MAX_TODO_CONTENT_LENGTH} characters")

        status = item["status"]
        if not isinstance(status, str) or status not in TODO_STATUSES:
            raise ValueError(
                f"todo at index {index} status must be one of "
                f"{', '.join(sorted(TODO_STATUSES))}"
            )
        if status == "in_progress":
            in_progress_count += 1

        priority = item["priority"]
        if not isinstance(priority, str) or priority not in TODO_PRIORITIES:
            raise ValueError(
                f"todo at index {index} priority must be one of "
                f"{', '.join(sorted(TODO_PRIORITIES))}"
            )

        validated.append(
            TodoItem(
                id=todo_id,
                content=content.strip(),
                status=status,
                priority=priority,
            )
        )

    if in_progress_count > 1:
        raise ValueError("todo list may contain at most one in_progress item")

    return validated


TODOWRITE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todowrite",
        "description": (
            "Create and maintain a structured task list for the current agent run. "
            "This replaces the entire todo list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": MAX_TODO_ITEMS,
                    "description": "The complete updated todo list for this run.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Stable todo item id."},
                            "content": {
                                "type": "string",
                                "description": "Specific actionable task description.",
                                "minLength": 1,
                                "maxLength": MAX_TODO_CONTENT_LENGTH,
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["id", "content", "status", "priority"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
    },
}

TODOREAD_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todoread",
        "description": "Read the current structured task list for this agent run.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


def builtin_todo_tool_schemas() -> list[dict[str, Any]]:
    """Return fresh provider-facing builtin todo tool schemas."""
    return [copy.deepcopy(TODOWRITE_TOOL_SCHEMA), copy.deepcopy(TODOREAD_TOOL_SCHEMA)]


def create_builtin_todo_tools(state: TodoState) -> dict[str, Callable[..., Any]]:
    """Create builtin callables bound to one run-local TodoState."""

    def todowrite(todos: Any = None, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise ValueError(
                f"todowrite received unexpected argument(s): {', '.join(sorted(kwargs))}"
            )
        if todos is None:
            raise ValueError("todowrite requires a todos argument")
        return state.replace(todos)

    def todoread(**kwargs: Any) -> dict[str, Any]:
        if kwargs:
            raise ValueError(
                f"todoread received unexpected argument(s): {', '.join(sorted(kwargs))}"
            )
        return state.snapshot()

    todowrite.__name__ = "todowrite"
    todoread.__name__ = "todoread"
    return {"todowrite": todowrite, "todoread": todoread}


def create_builtin_todo_bundle() -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Create schemas and callables with fresh state for one loop instance."""
    state = TodoState()
    return builtin_todo_tool_schemas(), create_builtin_todo_tools(state)
