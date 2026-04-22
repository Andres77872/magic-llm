"""TaskExecutor — ToolExecutor subclass for task/subagent runtime.

Handles BOTH task tools AND ordinary tools via routing:
- execute_async() checks if tool_call.name is in _task_registry
- If YES: use wrapped callable with depth/timeout/semaphore safeguards
- If NO: delegate to super().execute_async() (ordinary tool behavior)

This design enables ONE executor injection point with automatic fallback.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
import uuid
from typing import Any, Callable, Optional

from magic_llm.agent.types import (
    CanonicalToolCall,
    TaskError,
    TaskManifest,
    TaskResult,
    ToolResult,
)
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.normalizer import ResultNormalizer

logger = logging.getLogger(__name__)


# ContextVar for depth tracking (async task isolation)
# Each asyncio task has isolated context, ensuring per-execution isolation
TASK_DEPTH: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    'task_depth',
    default={}
)


class TaskExecutor(ToolExecutor):
    """ToolExecutor subclass that handles BOTH task tools AND ordinary tools.

    Routing Logic:
        - execute_async() checks if tool_call.name is in _task_registry
        - If YES: use wrapped callable with depth/timeout/semaphore safeguards
        - If NO: delegate to super().execute_async() (ordinary tool behavior)

    This design enables ONE executor injection point with automatic fallback,
    making it seamless for AsyncAgentLoop to use task subagents alongside
    regular tools (MCP, Fetch, etc.).

    Args:
        per_tool_timeout: Default timeout for ordinary tools (default: 30.0).
        enable_dedup: Enable fingerprint-based deduplication (default: False).
    """

    def __init__(
        self,
        per_tool_timeout: float = 30.0,
        enable_dedup: bool = False,
    ) -> None:
        super().__init__(per_tool_timeout=per_tool_timeout, enable_dedup=enable_dedup)
        # Task-specific registry (manifest + wrapped callable)
        self._task_registry: dict[str, TaskManifest] = {}
        # Per-task semaphores for concurrency control
        self._task_semaphores: dict[str, asyncio.Semaphore] = {}

    def register_task(
        self,
        manifest: TaskManifest,
        callable: Callable[..., Any],
    ) -> None:
        """Register a task/subagent with manifest and callable.

        Creates:
            - Semaphore(manifest.max_concurrency) for concurrency control
            - Wrapped callable with depth/timeout/semaphore safeguards
            - Entry in _task_registry for routing
            - Entry in base _registry for execution

        Args:
            manifest: TaskManifest with policy (id, timeout, concurrency, depth).
            callable: Async callable to execute (graph-side implementation from wrapper).

        Example:
            >>> executor = TaskExecutor()
            >>> manifest = TaskManifest(
            ...     id="web_search",
            ...     name="Web Search",
            ...     description="Search the web",
            ...     input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ...     timeout_seconds=30,
            ...     max_concurrency=5,
            ...     max_depth=3,
            ... )
            >>> async def search(query: str) -> dict:
            ...     return {"results": [...]}
            >>> executor.register_task(manifest, search)
        """
        # Create per-task semaphore for concurrency control
        semaphore = asyncio.Semaphore(manifest.max_concurrency)
        self._task_semaphores[manifest.id] = semaphore

        # Store manifest for routing and policy lookup
        self._task_registry[manifest.id] = manifest

        # Wrap callable with safeguards and register in base registry
        wrapped = self._wrap_task_callable(manifest, callable)
        super().register(manifest.id, wrapped)

        logger.debug(
            f"Registered task '{manifest.id}' with timeout={manifest.timeout_seconds}s, "
            f"max_concurrency={manifest.max_concurrency}, max_depth={manifest.max_depth}"
        )

    async def execute_async(self, tool_call: CanonicalToolCall) -> ToolResult:
        """Execute a tool call with routing logic.

        Routing:
            - If tool_call.name in _task_registry → wrapped callable (safeguards applied)
            - Else → super().execute_async() (ordinary tool behavior)

        Args:
            tool_call: CanonicalToolCall from LLM response.

        Returns:
            ToolResult with content as TaskResult JSON (for tasks) or plain string (for ordinary tools).
        """
        if tool_call.name in self._task_registry:
            # Task tool — use wrapped callable (safeguards already applied in registration)
            start = time.monotonic()

            try:
                # The wrapped callable returns TaskResult JSON string
                result_content = await self._registry[tool_call.name](**tool_call.arguments)
                duration_ms = (time.monotonic() - start) * 1000

                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=result_content,
                    is_error=False,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start) * 1000
                logger.error(f"TaskExecutor routing error for '{tool_call.name}': {exc}")
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=json.dumps({"error": str(exc), "type": type(exc).__name__}),
                    is_error=True,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    duration_ms=duration_ms,
                )
        else:
            # Ordinary tool — delegate to base executor (MCP, Fetch, etc.)
            return await super().execute_async(tool_call)

    def _wrap_task_callable(
        self,
        manifest: TaskManifest,
        callable: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Wrap callable with safeguards.

        Safeguards (in order):
            1. Generate unique task_id
            2. Validate input against manifest.input_schema (optional, soft validation)
            3. Check depth limit (ContextVar) — return early if exceeded
            4. Acquire semaphore
            5. Increment depth
            6. Apply timeout wrapper
            7. Execute callable
            8. Normalize result via ResultNormalizer
            9. Decrement depth (finally)
            10. Release semaphore (finally)

        Args:
            manifest: TaskManifest with policy.
            callable: Async callable from wrapper (graph-side implementation).

        Returns:
            Wrapped async callable that returns TaskResult JSON string.
        """

        async def wrapped(**kwargs: Any) -> str:
            task_id = uuid.uuid4().hex[:8]

            # Step 1: Check depth limit BEFORE acquiring semaphore
            # This prevents deadlock when depth is exceeded
            current_depth = _get_depth(manifest.id)
            if current_depth >= manifest.max_depth:
                result = TaskResult(
                    task_id=task_id,
                    task_type=manifest.id,
                    status="cancelled",
                    summary=(
                        f"## Task Cancelled\n\n"
                        f"Depth limit exceeded for '{manifest.name}'.\n\n"
                        f"Current depth: {current_depth}, max allowed: {manifest.max_depth}"
                    ),
                    error=TaskError(
                        error_type=TaskError.DEPTH_LIMIT,
                        message=f"Depth {current_depth} >= max_depth {manifest.max_depth}",
                        retryable=False,
                    ),
                )
                return result.to_tool_result_json()

            # Step 2: Acquire semaphore (queues if limit reached)
            async with self._task_semaphores[manifest.id]:
                # Step 3: Increment depth (before execution)
                _increment_depth(manifest.id)

                try:
                    # Step 4: Apply timeout and execute
                    raw_output = await asyncio.wait_for(
                        callable(**kwargs),
                        timeout=manifest.timeout_seconds,
                    )

                    # Step 5: Normalize result
                    result = ResultNormalizer.normalize(
                        raw_output=raw_output,
                        manifest=manifest,
                        task_id=task_id,
                        status="ok",
                    )

                except asyncio.TimeoutError:
                    result = TaskResult(
                        task_id=task_id,
                        task_type=manifest.id,
                        status="timeout",
                        summary=(
                            f"## Task Timeout\n\n"
                            f"Task '{manifest.name}' exceeded {manifest.timeout_seconds}s timeout."
                        ),
                        error=TaskError(
                            error_type=TaskError.TIMEOUT,
                            message=f"Timeout after {manifest.timeout_seconds}s",
                            retryable=True,
                        ),
                    )

                except json.JSONDecodeError as exc:
                    # Input validation error (from kwargs parsing edge case)
                    result = TaskResult(
                        task_id=task_id,
                        task_type=manifest.id,
                        status="failed",
                        summary=f"## Validation Error\n\n{exc.msg}",
                        error=TaskError(
                            error_type=TaskError.VALIDATION,
                            message=exc.msg,
                            retryable=False,
                        ),
                    )

                except Exception as exc:
                    # General execution error
                    error_msg = str(exc)
                    error_type_name = type(exc).__name__
                    logger.error(
                        f"Task '{manifest.id}' execution failed: {error_type_name}: {error_msg}"
                    )

                    result = TaskResult(
                        task_id=task_id,
                        task_type=manifest.id,
                        status="failed",
                        summary=f"## Task Failed\n\n{error_msg}",
                        error=TaskError(
                            error_type=TaskError.EXECUTION,
                            message=error_msg,
                            retryable=True,  # Most execution errors are retryable
                        ),
                    )

                finally:
                    # Step 6: Decrement depth (after execution, success or failure)
                    _decrement_depth(manifest.id)

                return result.to_tool_result_json()

        return wrapped

    def unregister_task(self, task_id: str) -> bool:
        """Remove a registered task by ID.

        Args:
            task_id: The task ID to remove.

        Returns:
            True if the task was found and removed, False otherwise.
        """
        if task_id in self._task_registry:
            del self._task_registry[task_id]
            self._task_semaphores.pop(task_id, None)
            # Also remove from base registry
            super().unregister(task_id)
            logger.debug(f"Unregistered task '{task_id}'")
            return True
        return False

    def get_registered_tasks(self) -> list[str]:
        """Get list of registered task IDs.

        Returns:
            List of task IDs currently registered.
        """
        return list(self._task_registry.keys())

    def get_task_manifest(self, task_id: str) -> Optional[TaskManifest]:
        """Get manifest for a specific task.

        Args:
            task_id: The task ID to look up.

        Returns:
            TaskManifest if found, None otherwise.
        """
        return self._task_registry.get(task_id)


# ─── ContextVar helpers for depth tracking ─────────────────────────────────────


def _get_depth(task_id: str) -> int:
    """Get current depth for a specific task_id.

    Args:
        task_id: The task ID to check.

    Returns:
        Current depth count for this task_id (0 if not set).
    """
    depths = TASK_DEPTH.get()
    return depths.get(task_id, 0)


def _increment_depth(task_id: str) -> int:
    """Increment depth for a specific task_id.

    Args:
        task_id: The task ID to increment.

    Returns:
        The new depth value after incrementing.
    """
    depths = TASK_DEPTH.get().copy()
    current = depths.get(task_id, 0)
    new_depth = current + 1
    depths[task_id] = new_depth
    TASK_DEPTH.set(depths)
    return new_depth


def _decrement_depth(task_id: str) -> int:
    """Decrement depth for a specific task_id.

    Args:
        task_id: The task ID to decrement.

    Returns:
        The new depth value after decrementing (min 0).
    """
    depths = TASK_DEPTH.get().copy()
    current = depths.get(task_id, 0)
    new_depth = max(0, current - 1)
    depths[task_id] = new_depth
    TASK_DEPTH.set(depths)
    return new_depth


def reset_depths() -> None:
    """Reset all depth counters.

    Used at execution start to ensure clean state.
    """
    TASK_DEPTH.set({})


def get_all_depths() -> dict[str, int]:
    """Get all current depth values.

    Returns:
        Dict of task_id -> depth for debugging/logging.
    """
    return TASK_DEPTH.get().copy()