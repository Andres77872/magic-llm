"""AgentHooks protocol — lifecycle callbacks for the agent loop.

Defines an observer-only protocol for hooking into the agent loop lifecycle.
All methods are optional (no-op default). Hooks MUST NOT modify loop state
or alter control flow — exceptions raised by hooks propagate to the caller
and terminate the loop.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from magic_llm.model import ModelChatResponse
from magic_llm.agent.types import AgentState, ToolResult


@runtime_checkable
class AgentHooks(Protocol):
    """Lifecycle callbacks for the agent loop.

    All methods are optional — the default implementation is a no-op.
    Hooks are OBSERVERS only: they MUST NOT modify loop state or alter
    control flow. Exceptions raised by hooks propagate to the caller
    and terminate the loop.

    Methods:
        on_iteration_start: Called before each LLM call.
        on_llm_response: Called after each LLM response, before tool extraction.
        on_tool_start: Called before each tool execution.
        on_tool_complete: Called after each tool execution (success or error).
        on_loop_complete: Called after the loop exits (NORMAL exit only — NOT budget-exceeded).
        on_budget_exceeded: Called when a budget constraint is violated.
    """

    def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        """Called before each LLM call in the loop.

        Args:
            iteration: The current 0-indexed iteration number.
            state: The current agent state (read-only).
        """
        ...

    def on_llm_response(
        self, response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after each LLM response, before tool extraction.

        Args:
            response: The raw LLM response.
            state: The current agent state (read-only).
        """
        ...

    def on_tool_start(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        state: AgentState,
    ) -> None:
        """Called before each tool execution.

        Args:
            tool_name: The name of the tool about to be executed.
            tool_call_id: The provider-specific tool call identifier.
            arguments: The parsed tool arguments.
            state: The current agent state (read-only).
        """
        ...

    def on_tool_complete(self, result: ToolResult, state: AgentState) -> None:
        """Called after each tool execution (success or error).

        Args:
            result: The structured tool execution result.
            state: The current agent state (read-only).
        """
        ...

    def on_loop_complete(
        self, final_response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after the loop exits (NORMAL exit only — NOT budget-exceeded).

        For budget-exceeded exits, on_budget_exceeded fires instead.
        The loop does NOT call both on_loop_complete AND on_budget_exceeded
        for the same exit.

        Args:
            final_response: The final LLM response.
            state: The final agent state (read-only).
        """
        ...

    def on_budget_exceeded(self, budget_type: str, details: str) -> None:
        """Called when a budget constraint is violated.

        Args:
            budget_type: Which budget limit was exceeded.
            details: Human-readable details about the breach.
        """
        ...
