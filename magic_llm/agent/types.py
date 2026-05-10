"""Core data models, exceptions, and types for the agent loop.

This module defines the foundational types used across the agent package:
- ToolResult: Structured result from a single tool execution (Pydantic model).
- CanonicalToolCall: Provider-agnostic representation of a tool call request.
- AgentBudget: Execution bounds for the agent loop.
- AgentState: Explicit typed state for the agent loop (read-only from consumer perspective).
- AgentLoopError: Base exception for agent loop errors.
- AgentBudgetExceeded: Raised when a budget constraint is violated.
- ToolExecutionError: Raised when a tool execution fails critically.

Task subagent types (for runtime contract):
- TaskManifest: Machine-readable subagent identity, schema, and policy.
- TaskResult: Structured output envelope with status, summary, error.
- TaskError: Error taxonomy for task failures.
- TaskBudget: AgentBudget extended with max_depth field.
- TaskState: AgentState extended with task_depths for observability.

Global Depth API (re-exported from _loop_shared and config for public convenience):
- GLOBAL_DEPTH: ContextVar for global nesting depth tracking.
- MAX_GLOBAL_DEPTH: Configurable constant for maximum global depth (default: 10).
- get_global_depth: Helper to get current global depth value.
- increment_global_depth: Helper to increment global depth.
- decrement_global_depth: Helper to decrement global depth.
- reset_global_depth: Helper to reset global depth to 0.

NOTE: These are RE-EXPORTS from _loop_shared and config. The actual runtime objects
are defined there, ensuring consumers of types.py observe the same state as TaskExecutor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field


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
        max_iterations: Hard cap on loop iterations (default: 150).
        max_input_tokens: Max cumulative input tokens across the loop (None = no limit).
        max_output_tokens: Max cumulative output tokens across the loop (None = no limit).
        wall_clock_timeout: Max wall-clock seconds for the entire loop (None = no limit).
    """

    max_iterations: int = 150
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


# ─── Task Subagent Runtime Contract Types ─────────────────────────────────────


class TaskManifest(BaseModel):
    """Runtime manifest for task/subagent execution.

    Passed to MagicLLM.register_task() by wrappers like magic-agents.
    Contains only fields needed for execution safeguards (runtime subset).

    This is distinct from SubagentManifest in magic-agents, which contains
    YAML-specific fields like apiVersion, kind, source_file.

    Attributes:
        id: Stable registry ID (matches tool name).
        name: Human-readable name for display/logging.
        description: When-to-use summary for routing.
        input_schema: JSON Schema for input validation.
        timeout_seconds: Per-task timeout (default: 30s).
        max_concurrency: Concurrent instances allowed (default: 5).
        max_depth: Recursion depth limit (default: 3).

    Nested LLM Node Configuration (optional):
        nested_tools: Child's own tools (explicit list, NOT inherited from parent).
        nested_budget: Child's token/iteration limits (independent from parent budget).
        nested_model_override: Child's model name (else uses parent's model).
        budget_cascade: When True, child inherits remaining parent budget; when False, child uses own budget.

    Example:
        >>> manifest = TaskManifest(
        ...     id="research_task",
        ...     name="Research Task",
        ...     description="Search the web for information",
        ...     input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ...     nested_tools=[web_search, web_fetch],  # Child's own tools
        ...     nested_budget=AgentBudget(max_iterations=5),
        ...     budget_cascade=False,  # Child uses own budget
        ... )
    """

    id: str = Field(..., pattern=r'^[a-z0-9._-]+$')
    name: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_concurrency: int = Field(default=5, ge=1, le=20)
    max_depth: int = Field(default=3, ge=1, le=10)

    # Nested LLM node configuration (optional)
    # When nested_tools is provided, the task is treated as a native nested LLM node
    # When nested_tools is None, the task behaves as a wrapped callable (current behavior)
    nested_tools: Optional[list[Any]] = None  # Child's own tools (explicit, no inheritance)
    nested_budget: Optional[AgentBudget] = None  # Child's token/iteration limits
    nested_model_override: Optional[str] = None  # Child's model name (else parent model)
    budget_cascade: bool = False  # Child inherits remaining parent budget when True


class TaskError(BaseModel):
    """Structured error representation for TaskResult.

    Every task failure produces explicit classification.
    Never silently swallow exceptions.

    Attributes:
        error_type: Classification constant (ValidationError, TimeoutError, etc.).
        message: Human-readable error description.
        retryable: Whether retry might succeed (default: False).
    """

    error_type: str
    message: str
    retryable: bool = False

    # Classification constants (ClassVar for Pydantic compatibility)
    VALIDATION: ClassVar[str] = "ValidationError"
    TIMEOUT: ClassVar[str] = "TimeoutError"
    EXECUTION: ClassVar[str] = "ExecutionError"
    DEPTH_LIMIT: ClassVar[str] = "DepthLimitError"


class TaskResult(BaseModel):
    """Structured output envelope for task/subagent execution.

    Main return surface is plain-text Markdown summary.
    Serialized as ToolResult.content for parent LLM injection.

    Attributes:
        task_id: Unique invocation ID (UUID-like).
        task_type: Matches TaskManifest.id for correlation.
        status: Execution outcome (ok, failed, timeout, cancelled).
        summary: Plain-text Markdown for parent LLM consumption.
        error: Optional TaskError when status != "ok".
    """

    task_id: str
    task_type: str
    status: Literal["ok", "failed", "timeout", "cancelled"]
    summary: str
    error: Optional[TaskError] = None

    def to_tool_result_json(self) -> str:
        """Serialize for ToolResult.content injection.

        Returns minimal JSON with summary as primary field.
        Parent LLM receives summary directly in role="tool" message.
        """
        return self.model_dump_json(exclude_none=True)


@dataclass
class TaskBudget(AgentBudget):
    """AgentBudget extended with task-specific limits.

    Attributes:
        max_depth: Global depth cap for all tasks (default: 10).
    """

    max_depth: int = 10


@dataclass
class TaskState(AgentState):
    """AgentState extended with task depth tracking.

    NOTE: ContextVar is used for async task isolation.
    TaskState.task_depths is for observability/debugging only.

    Attributes:
        task_depths: Dict of task_id -> current depth (observability).
    """

    task_depths: dict[str, int] = field(default_factory=dict)


# ─── Exceptions ───────────────────────────────────────────────────────────────


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


# ─── Global Depth API Re-Exports (from _loop_shared and config) ─────────────────
# These are RE-EXPORTS, not new definitions. The actual runtime objects are in
# _loop_shared.py (GLOBAL_DEPTH, helpers) and config.py (MAX_GLOBAL_DEPTH).
# This ensures consumers of types.py observe the same state as TaskExecutor.

from magic_llm.agent._loop_shared import (
    GLOBAL_DEPTH,
    get_global_depth,
    increment_global_depth,
    decrement_global_depth,
    reset_global_depth,
)

from magic_llm.agent.config import MAX_GLOBAL_DEPTH
