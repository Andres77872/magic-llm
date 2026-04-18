"""Shared loop helpers used by both AgentLoop and AsyncAgentLoop.

This module contains ONLY synchronous, pure/stateless helper functions.
NO async def, NO await, NO asyncio imports — both sync and async loops
depend on these helpers without contamination.

Functions:
    _check_budget: Validates iteration, wall-clock, and token budgets.
    _build_initial_chat: Creates a ModelChat from system/user/extra messages.
    _register_tools_with_executor: Registers tools from callables and dicts.
    _finalize_response: Concatenates collected content into the final response.
    _invoke_hook_safely: Calls a hook method, propagating exceptions.
    _compute_fingerprint: SHA-256 hash for tool deduplication.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Optional

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
)
from magic_llm.agent.tool_executor import ToolExecutor


def _check_budget(
    state: AgentState,
    budget: AgentBudget,
    *,
    include_tokens: bool = False,
) -> None:
    """Check budget constraints and raise AgentBudgetExceeded if breached.

    Args:
        state: Current agent state with step, tokens, and start_time.
        budget: Budget configuration with limits.
        include_tokens: When False, only checks iterations and wall-clock
            (pre-LLM call). When True, also checks token budgets (post-LLM).

    Raises:
        AgentBudgetExceeded: If any budget limit is exceeded.
    """
    # Iteration check — always checked (cheap, no response needed)
    if state.step >= budget.max_iterations:
        raise AgentBudgetExceeded(
            budget_type="max_iterations",
            limit=budget.max_iterations,
            current=state.step,
        )

    # Wall-clock check — always checked (time comparison only)
    if budget.wall_clock_timeout is not None and state.start_time is not None:
        elapsed = time.monotonic() - state.start_time
        if elapsed >= budget.wall_clock_timeout:
            raise AgentBudgetExceeded(
                budget_type="wall_clock_timeout",
                limit=budget.wall_clock_timeout,
                current=round(elapsed, 3),
            )

    # Token checks — only after LLM response provides usage data
    if include_tokens:
        if budget.max_input_tokens is not None:
            if state.total_input_tokens >= budget.max_input_tokens:
                raise AgentBudgetExceeded(
                    budget_type="max_input_tokens",
                    limit=budget.max_input_tokens,
                    current=state.total_input_tokens,
                )

        if budget.max_output_tokens is not None:
            if state.total_output_tokens >= budget.max_output_tokens:
                raise AgentBudgetExceeded(
                    budget_type="max_output_tokens",
                    limit=budget.max_output_tokens,
                    current=state.total_output_tokens,
                )


def _build_initial_chat(
    user_input: str,
    system_prompt: Optional[str] = None,
    extra_messages: Optional[list[dict[str, Any]]] = None,
) -> ModelChat:
    """Build the initial ModelChat with system, extra, and user messages.

    Args:
        user_input: The primary user message to start the conversation.
        system_prompt: Optional system prompt (added as first message).
        extra_messages: Optional list of message dicts to insert before user_input.

    Returns:
        A ModelChat instance with the initial conversation history.
    """
    chat = ModelChat(system=system_prompt)

    # Add extra messages before the user input
    if extra_messages:
        for msg in extra_messages:
            chat.add_message(msg.get("role", "user"), msg.get("content", ""))

    # Add the user input
    chat.add_user_message(user_input)

    return chat


def _register_tools_with_executor(
    executor: ToolExecutor,
    tools: Optional[list[Any]] = None,
    tool_functions: Optional[dict[str, Callable[..., Any]]] = None,
) -> None:
    """Register tools with the executor from both callables and dict sources.

    Args:
        executor: The ToolExecutor to register tools with.
        tools: List of callable tools (registered by __name__) or dict tool
            specs (name extracted and resolved from tool_functions).
        tool_functions: Dict mapping custom names to callables.
    """
    # Register callable tools by their __name__
    if tools:
        for tool in tools:
            if callable(tool) and not isinstance(tool, dict):
                name = getattr(tool, "__name__", None)
                if name:
                    executor.register(name, tool)
            elif isinstance(tool, dict):
                # Dict tool spec: extract name and resolve from tool_functions
                fn_def = tool.get("function") if tool.get("type") == "function" else tool
                name = fn_def.get("name") if isinstance(fn_def, dict) else None
                if name and tool_functions and callable(tool_functions.get(name)):
                    executor.register(name, tool_functions[name])

    # Register tools from dict (custom name → callable)
    if tool_functions:
        for name, fn in tool_functions.items():
            if callable(fn):
                executor.register(name, fn)


def _finalize_response(
    response: ModelChatResponse,
    collected_content: list[str],
    separator: str = "\n\n",
) -> ModelChatResponse:
    """Concatenate collected content into the final response.

    Mutates the response in-place when content exists and choices are present.

    Args:
        response: The last LLM response to finalize.
        collected_content: List of text content accumulated across iterations.
        separator: String to join content pieces with.

    Returns:
        The (possibly mutated) response.
    """
    if not collected_content:
        return response

    final_content = separator.join(collected_content)

    # Only mutate if there are choices to mutate
    if response.choices:
        response.choices[0].message.content = final_content

    return response


def _invoke_hook_safely(
    hook_method: Optional[Callable[..., Any]],
    *args: Any,
) -> None:
    """Invoke a hook method safely, propagating any exceptions.

    "Safely" means: if the hook is None, skip. Otherwise call it and
    let any exception propagate (hooks are observers, not interceptors).

    Args:
        hook_method: The hook method to call, or None to skip.
        *args: Arguments to pass to the hook method.
    """
    if hook_method is None:
        return
    hook_method(*args)


def _compute_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Compute a SHA-256 fingerprint for a tool call (deduplication).

    Matches the spec formula exactly:
        hashlib.sha256(json.dumps({"name": name, "args": args}, sort_keys=True)).hexdigest()

    Args:
        name: The tool name.
        arguments: The parsed arguments dict.

    Returns:
        A hex digest string.
    """
    try:
        payload = json.dumps({"name": name, "args": arguments}, sort_keys=True)
    except (TypeError, ValueError):
        # Fallback for non-serializable args
        payload = json.dumps(
            {"name": name, "args": str(arguments)}, sort_keys=True
        )
    return hashlib.sha256(payload.encode()).hexdigest()
