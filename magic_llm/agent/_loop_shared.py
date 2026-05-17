"""Shared loop helpers used by both AgentLoop and AsyncAgentLoop.

This module contains ONLY synchronous, pure/stateless helper functions.
NO async def, NO await, NO asyncio imports — both sync and async loops
depend on these helpers without contamination.

Functions:
    _check_budget: Validates iteration, wall-clock, and token budgets.
    _build_initial_chat: Creates a ModelChat from system/user/extra messages.
    _register_tools_with_executor: Registers tools from callables and dicts.
    _finalize_response: Concatenates collected content into the final response.
    _invoke_hook_safely: Calls a hook method, isolating exceptions (logs, does not propagate).
    _compute_fingerprint: SHA-256 hash for tool deduplication.

Global Depth Helpers:
    get_global_depth: Get current global nesting depth.
    increment_global_depth: Increment global depth counter.
    decrement_global_depth: Decrement global depth counter (min 0).
    reset_global_depth: Reset global depth to 0.

Parent Context ContextVars:
    PARENT_BUDGET: ContextVar for parent AgentBudget (cascade computation).
    PARENT_STATE: ContextVar for parent AgentState (cascade computation).

Budget Cascade Helper:
    _compute_child_budget: Compute child budget based on cascade policy.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
    TaskManifest,
)
from magic_llm.agent.tool_executor import ToolExecutor

if TYPE_CHECKING:
    from magic_llm.agent.hooks import AgentHooks


logger = logging.getLogger(__name__)


# ─── Global Depth ContextVar ───────────────────────────────────────────────────
# Tracks total nesting depth across all task IDs (safety cap for unbounded nesting).
# Independent from TASK_DEPTH (per-task-id tracking) in task_executor.py.

GLOBAL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    'global_depth',
    default=0
)


# ─── Parent Context ContextVars ─────────────────────────────────────────────────
# For budget cascade computation: child needs to know parent's remaining budget.

PARENT_BUDGET: contextvars.ContextVar[Optional[AgentBudget]] = contextvars.ContextVar(
    'parent_budget',
    default=None
)

PARENT_STATE: contextvars.ContextVar[Optional[AgentState]] = contextvars.ContextVar(
    'parent_state',
    default=None
)

PARENT_HOOKS: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    'parent_hooks',
    default=None
)
"""ContextVar for parent AgentHooks (nested LLM node hook propagation).

Set by AsyncAgentLoop.run()/stream() before child loop creation.
Read by TaskExecutor._execute_nested_llm_node() to propagate hooks
to child loops. Reset in try/finally blocks to prevent cross-run
contamination.

Follows the exact PARENT_BUDGET/PARENT_STATE pattern.
"""

DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    'depth',
    default=0
)
"""ContextVar for auto-incrementing nesting depth.

Incremented in TaskExecutor._execute_nested_llm_node() before
child loop creation. Used by NestedHookRelay to tag child events
with the current depth level.

Root = 0, first child = 1, grandchild = 2, etc.
"""


# ─── Global Depth Helper Functions ──────────────────────────────────────────────


def get_global_depth() -> int:
    """Get current global nesting depth.

    Returns:
        Current global depth value (0 if not set).
    """
    return GLOBAL_DEPTH.get()


def increment_global_depth() -> int:
    """Increment global depth counter.

    Returns:
        The new depth value after incrementing.
    """
    current = GLOBAL_DEPTH.get()
    new_depth = current + 1
    GLOBAL_DEPTH.set(new_depth)
    return new_depth


def decrement_global_depth() -> int:
    """Decrement global depth counter.

    Returns:
        The new depth value after decrementing (min 0).
    """
    current = GLOBAL_DEPTH.get()
    new_depth = max(0, current - 1)
    GLOBAL_DEPTH.set(new_depth)
    return new_depth


def reset_global_depth() -> None:
    """Reset global depth to 0.

    Used at execution start to ensure clean state.
    """
    GLOBAL_DEPTH.set(0)


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


def _clone_chat(chat: ModelChat) -> ModelChat:
    """Clone a ModelChat so loop mutations do not affect the caller."""
    cloned = ModelChat(
        system=None,
        max_input_tokens=chat.max_input_tokens,
        extra_args=copy.deepcopy(chat.extra_args),
    )
    cloned.messages = copy.deepcopy(chat.messages)
    return cloned


def _build_initial_chat(
    user_input: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_messages: Optional[list[dict[str, Any]]] = None,
    initial_chat: Optional[ModelChat] = None,
) -> ModelChat:
    """Build the initial ModelChat with system, extra, and user messages.

    When both ``system_prompt`` and ``initial_chat`` are provided, the
    ``system_prompt`` is prepended to the first existing system message
    in the cloned chat (or inserted at position 0 if none exists). This
    ensures that prompt_fragment (resolved and merged into system_prompt
    by the caller) is not lost when a prebuilt chat is used.

    There is NO deduplication: if ``system_prompt`` content already appears
    in ``initial_chat``, it will appear twice. Callers are responsible for
    ensuring they do not pass duplicate content.

    Args:
        user_input: The primary user message to start the conversation.
        system_prompt: Optional system prompt (added as first message).
        extra_messages: Optional list of message dicts to insert before user_input.
        initial_chat: Optional prebuilt chat history. When provided, it is
            cloned and returned with system_prompt merged in.

    Returns:
        A ModelChat instance with the initial conversation history.
    """
    if initial_chat is not None:
        cloned = _clone_chat(initial_chat)
        if system_prompt:
            # Find first existing system message to merge into
            sys_idx = None
            for i, msg in enumerate(cloned.messages):
                if msg.get("role") == "system":
                    sys_idx = i
                    break
            if sys_idx is not None:
                # Prepend system_prompt (containing resolved prompt_fragment)
                # to existing system message content.
                cloned.messages[sys_idx]["content"] = (
                    f"{system_prompt}\n\n{cloned.messages[sys_idx]['content']}"
                ).strip()
            else:
                # No existing system message — insert at position 0
                cloned.messages.insert(0, {
                    "role": "system",
                    "content": system_prompt,
                })
        return cloned

    if user_input is None:
        raise ValueError("user_input is required when initial_chat is not provided")

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
    state: Optional[AgentState] = None,
) -> None:
    """Invoke a hook method safely, isolating exceptions (logs, does not propagate).

    Hook exceptions are caught and logged with full context, then execution
    continues. This ensures hook failures do NOT terminate the agent loop.

    Args:
        hook_method: The hook method to call, or None to skip.
        *args: Arguments to pass to the hook method.
        state: Optional AgentState for logging context (step, messages count, etc).
    """
    if hook_method is None:
        return
    
    hook_name = getattr(hook_method, "__name__", "unknown_hook")
    
    try:
        hook_method(*args)
    except Exception as e:
        # Log hook failure with full context
        timestamp = datetime.now(timezone.utc).isoformat()
        error_type = type(e).__name__
        error_message = str(e)
        
        # Build state context for logging
        state_context = {}
        if state is not None:
            state_context = {
                "step": state.step,
                "messages_count": len(state.messages) if state.messages else 0,
                "total_input_tokens": state.total_input_tokens,
                "total_output_tokens": state.total_output_tokens,
            }
        
        logger.warning(
            "Hook '%s' raised exception: %s: %s",
            hook_name,
            error_type,
            error_message,
            extra={
                "timestamp": timestamp,
                "hook_name": hook_name,
                "error_type": error_type,
                "error_message": error_message,
                "state": state_context,
            }
        )


def _compute_child_budget(
    manifest: TaskManifest,
    parent_budget: Optional[AgentBudget],
    parent_state: Optional[AgentState],
) -> AgentBudget:
    """Compute effective child budget based on cascade policy.

    Cascade logic:
        1. budget_cascade=True + nested_budget provided → intersection: min(child_limit, parent_remaining)
        2. budget_cascade=True + nested_budget=None → use parent remaining directly
        3. budget_cascade=False + nested_budget provided → use child budget exclusively
        4. budget_cascade=False + nested_budget=None → return AgentBudget() (default)

    Parent remaining computation:
        - max_iterations: parent_budget.max_iterations - parent_state.step
        - wall_clock_timeout: parent_budget.wall_clock_timeout - elapsed_time
        - max_input_tokens: parent_budget.max_input_tokens - parent_state.total_input_tokens
        - max_output_tokens: parent_budget.max_output_tokens - parent_state.total_output_tokens

    Args:
        manifest: TaskManifest with nested_budget and budget_cascade flag.
        parent_budget: Parent's configured budget (max_iterations, wall_clock, etc).
        parent_state: Parent's current state (step count, elapsed time).

    Returns:
        AgentBudget for child loop (cascade or override).
    """
    # No parent context → standalone execution (use default or nested_budget)
    if parent_budget is None or parent_state is None:
        if manifest.nested_budget is not None:
            return AgentBudget(
                max_iterations=manifest.nested_budget.max_iterations,
                max_input_tokens=manifest.nested_budget.max_input_tokens,
                max_output_tokens=manifest.nested_budget.max_output_tokens,
                wall_clock_timeout=manifest.nested_budget.wall_clock_timeout,
            )
        return AgentBudget()

    # Compute parent remaining budget
    parent_iterations_remaining = parent_budget.max_iterations - parent_state.step

    parent_wall_clock_remaining = None
    if parent_budget.wall_clock_timeout is not None and parent_state.start_time is not None:
        elapsed = time.monotonic() - parent_state.start_time
        parent_wall_clock_remaining = max(0.0, parent_budget.wall_clock_timeout - elapsed)

    parent_input_tokens_remaining = None
    if parent_budget.max_input_tokens is not None:
        parent_input_tokens_remaining = max(0, parent_budget.max_input_tokens - parent_state.total_input_tokens)

    parent_output_tokens_remaining = None
    if parent_budget.max_output_tokens is not None:
        parent_output_tokens_remaining = max(0, parent_budget.max_output_tokens - parent_state.total_output_tokens)

    # Apply cascade logic
    if manifest.budget_cascade:
        # Cascade enabled: child inherits parent remaining budget (with intersection if nested_budget)
        if manifest.nested_budget is not None:
            # Intersection: min(child config, parent remaining)
            return AgentBudget(
                max_iterations=min(manifest.nested_budget.max_iterations, parent_iterations_remaining),
                max_input_tokens=(
                    min(manifest.nested_budget.max_input_tokens or 0, parent_input_tokens_remaining or 0)
                    if manifest.nested_budget.max_input_tokens is not None and parent_input_tokens_remaining is not None
                    else manifest.nested_budget.max_input_tokens
                ),
                max_output_tokens=(
                    min(manifest.nested_budget.max_output_tokens or 0, parent_output_tokens_remaining or 0)
                    if manifest.nested_budget.max_output_tokens is not None and parent_output_tokens_remaining is not None
                    else manifest.nested_budget.max_output_tokens
                ),
                wall_clock_timeout=(
                    min(manifest.nested_budget.wall_clock_timeout or 0.0, parent_wall_clock_remaining or 0.0)
                    if manifest.nested_budget.wall_clock_timeout is not None and parent_wall_clock_remaining is not None
                    else manifest.nested_budget.wall_clock_timeout
                ),
            )
        else:
            # No nested_budget: use parent remaining directly
            return AgentBudget(
                max_iterations=parent_iterations_remaining,
                max_input_tokens=parent_input_tokens_remaining,
                max_output_tokens=parent_output_tokens_remaining,
                wall_clock_timeout=parent_wall_clock_remaining,
            )
    else:
        # Cascade disabled: use child's own budget or default
        if manifest.nested_budget is not None:
            return AgentBudget(
                max_iterations=manifest.nested_budget.max_iterations,
                max_input_tokens=manifest.nested_budget.max_input_tokens,
                max_output_tokens=manifest.nested_budget.max_output_tokens,
                wall_clock_timeout=manifest.nested_budget.wall_clock_timeout,
            )
        return AgentBudget()


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
