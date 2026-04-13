"""Core data models, exceptions, and types for the agent loop.

This module defines the foundational types used across the agent package:
- ToolResult: Structured result from a single tool execution (Pydantic model).
- CanonicalToolCall: Provider-agnostic representation of a tool call request.
- AgentBudget: Execution bounds for the agent loop.
- AgentState: Explicit typed state for the agent loop (read-only from consumer perspective).
- AgentLoopError: Base exception for agent loop errors.
- AgentBudgetExceeded: Raised when a budget constraint is violated.
- ToolExecutionError: Raised when a tool execution fails critically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Structured result from a single tool execution.

    Attributes:
        tool_call_id: Provider-specific tool call identifier (e.g. 'call_abc123', 'toolu_01XYZ').
        name: The tool name.
        content: Serialized output content (JSON string for structured data, plain text otherwise).
        is_error: Whether the tool execution resulted in an error.
        error: Error message (populated when is_error=True).
        error_type: The exception class name (e.g. 'ValueError', 'TimeoutError').
        duration_ms: Execution time in milliseconds.
        is_deduplicated: Whether this result was returned from the deduplication cache.
    """

    tool_call_id: Optional[str] = None
    name: str
    content: str
    is_error: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    duration_ms: float = 0.0
    is_deduplicated: bool = False


class CanonicalToolCall:
    """Provider-agnostic representation of a tool call request.

    Uses __slots__ for memory efficiency since many tool calls may be created
    during a single loop run.

    Attributes:
        id: The provider-specific tool call identifier.
        name: The tool/function name.
        arguments: Parsed arguments as a dict (never a JSON string).
    """

    __slots__ = ("id", "name", "arguments")

    def __init__(self, id: str, name: str, arguments: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments

    def __repr__(self) -> str:
        return f"CanonicalToolCall(id={self.id!r}, name={self.name!r}, arguments={self.arguments!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonicalToolCall):
            return NotImplemented
        return (
            self.id == other.id
            and self.name == other.name
            and self.arguments == other.arguments
        )

    def __hash__(self) -> int:
        return hash((self.id, self.name, frozenset(self.arguments.items())))


@dataclass
class AgentBudget:
    """Execution bounds for the agent loop.

    Attributes:
        max_iterations: Hard cap on loop iterations (default: 10).
        max_input_tokens: Max cumulative input tokens across the loop (None = no limit).
        max_output_tokens: Max cumulative output tokens across the loop (None = no limit).
        wall_clock_timeout: Max wall-clock seconds for the entire loop (None = no limit).
    """

    max_iterations: int = 10
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    wall_clock_timeout: Optional[float] = None


@dataclass
class AgentState:
    """Explicit typed state for the agent loop.

    Read-only from consumer perspective — mutations to a returned state
    copy do NOT affect the internal loop state.

    Attributes:
        messages: The full conversation history (internal canonical format).
        step: Current iteration number (0-indexed).
        total_input_tokens: Cumulative input tokens consumed.
        total_output_tokens: Cumulative output tokens produced.
        executed_fingerprints: Set of tool call fingerprints already executed
            (only populated when deduplicate=True).
        start_time: Wall-clock start timestamp (for timeout tracking).
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    executed_fingerprints: set[str] = field(default_factory=set)
    start_time: Optional[float] = None


class AgentLoopError(Exception):
    """Base exception for agent loop errors."""

    pass


class AgentBudgetExceeded(AgentLoopError):
    """Raised when a budget constraint is violated.

    Attributes:
        budget_type: Which budget limit was exceeded (e.g. 'max_iterations', 'wall_clock_timeout').
        limit: The configured limit value.
        current: The current value at the time of breach.
    """

    def __init__(self, budget_type: str, limit: Any, current: Any) -> None:
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"Budget exceeded: {budget_type} limit={limit}, current={current}"
        )


class ToolExecutionError(AgentLoopError):
    """Raised when a tool execution fails critically.

    Attributes:
        tool_name: The name of the tool that failed.
        error: The error message.
    """

    def __init__(self, tool_name: str, error: str) -> None:
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"Tool '{tool_name}' failed: {error}")
