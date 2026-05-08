"""ToolExecutor — isolated tool execution with parallel support and structured results.

This module provides the ToolExecutor class responsible for:
- Tool registration and lookup
- Synchronous and parallel execution via concurrent.futures
- Async execution variants (execute_async, execute_parallel_async)
- Per-tool timeout enforcement
- Fingerprint-based deduplication (opt-in)
- Structured error capture (ToolResult with is_error, error, error_type)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.util import is_async_callable


class ToolExecutor:
    """Isolated tool execution with parallel support and structured results.

    Args:
        per_tool_timeout: Default timeout in seconds for each tool execution (default: 30.0).
        enable_dedup: Enable fingerprint-based deduplication (default: False).
        max_content_size: Global max content size in characters (default: 50000).
            Content exceeding this limit is truncated with a [TRUNCATED] suffix.
        max_content_sizes: Per-tool-name max content size overrides.
            Key is tool name, value is max chars. Takes precedence over global.
        tool_timeouts: Per-tool-name timeout overrides in seconds.
            Key is tool name, value is timeout. Takes precedence over per_tool_timeout.
    """

    def __init__(
        self,
        per_tool_timeout: float = 30.0,
        enable_dedup: bool = False,
        max_content_size: int = 50000,
        max_content_sizes: dict[str, int] | None = None,
        tool_timeouts: dict[str, float] | None = None,
    ) -> None:
        self._per_tool_timeout = per_tool_timeout
        self._enable_dedup = enable_dedup
        self._max_content_size = max_content_size
        self._max_content_sizes = max_content_sizes or {}
        self._tool_timeouts = tool_timeouts or {}
        self._registry: dict[str, Callable[..., Any]] = {}
        self._dedup_cache: dict[str, ToolResult] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a tool callable under the given name.

        Args:
            name: The tool name used for lookup during execution.
            fn: The callable to invoke when the tool is executed.
        """
        self._registry[name] = fn

    def unregister(self, name: str) -> bool:
        """Remove a registered tool by name.

        Args:
            name: The tool name to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        if name in self._registry:
            del self._registry[name]
            self._dedup_cache.pop(name, None)
            return True
        return False

    def register_many(self, tools: list[tuple[str, Callable[..., Any]]]) -> None:
        """Register multiple tools at once.

        Args:
            tools: A list of (name, fn) tuples.
        """
        for name, fn in tools:
            self.register(name, fn)

    def execute(self, tool_call: CanonicalToolCall) -> ToolResult:
        """Execute a single tool call synchronously.

        Args:
            tool_call: The canonical tool call to execute.

        Returns:
            A ToolResult with the execution outcome (success or error).
        """
        # Check dedup cache
        if self._enable_dedup:
            fingerprint = self._compute_fingerprint(
                tool_call.name, tool_call.arguments
            )
            if fingerprint in self._dedup_cache:
                cached = self._dedup_cache[fingerprint].model_copy()
                cached.is_deduplicated = True
                return cached

        start = time.monotonic()

        # Look up tool
        fn = self._registry.get(tool_call.name)
        if fn is None:
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                is_error=True,
                error=f"Unknown tool: {tool_call.name}",
                error_type="UnknownToolError",
                duration_ms=duration_ms,
            )
            return result

        # Execute with timeout (per-tool override supported)
        effective_timeout = self._resolve_timeout(tool_call.name)
        try:
            output = self._execute_with_timeout(
                fn, tool_call.arguments, timeout=effective_timeout
            )
        except FuturesTimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                is_error=True,
                error=f"Tool '{tool_call.name}' timed out after {effective_timeout}s",
                error_type="TimeoutError",
                duration_ms=duration_ms,
            )
            return result
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            error_msg = str(exc)
            error_type = type(exc).__name__
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=json.dumps({"error": error_msg, "type": error_type}),
                is_error=True,
                error=error_msg,
                error_type=error_type,
                duration_ms=duration_ms,
            )
            return result

        duration_ms = (time.monotonic() - start) * 1000

        # Serialize output (with per-tool max_content_size enforcement)
        content = self._serialize_output(output, tool_name=tool_call.name)
        truncated = content.endswith("[TRUNCATED]")

        result = ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=truncated,
            duration_ms=duration_ms,
        )

        # Cache for dedup
        if self._enable_dedup:
            self._dedup_cache[fingerprint] = result

        return result

    def execute_parallel(
        self, tool_calls: list[CanonicalToolCall]
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel using ThreadPoolExecutor.

        Results are returned in the same order as the input calls.

        Args:
            tool_calls: The list of canonical tool calls to execute.

        Returns:
            A list of ToolResult objects in input order.
        """
        if not tool_calls:
            return []

        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            results = list(executor.map(self.execute, tool_calls))
        return results

    async def execute_async(self, tool_call: CanonicalToolCall) -> ToolResult:
        """Execute a single tool call, supporting both sync and async callables.

        If the callable is async, it is awaited directly. If sync, it runs
        in a thread executor to avoid blocking the event loop.

        Args:
            tool_call: The canonical tool call to execute.

        Returns:
            A ToolResult with the execution outcome.
        """
        # Check dedup cache
        if self._enable_dedup:
            fingerprint = self._compute_fingerprint(
                tool_call.name, tool_call.arguments
            )
            if fingerprint in self._dedup_cache:
                cached = self._dedup_cache[fingerprint].model_copy()
                cached.is_deduplicated = True
                return cached

        start = time.monotonic()

        # Look up tool
        fn = self._registry.get(tool_call.name)
        if fn is None:
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                is_error=True,
                error=f"Unknown tool: {tool_call.name}",
                error_type="UnknownToolError",
                duration_ms=duration_ms,
            )
            return result

        effective_timeout = self._resolve_timeout(tool_call.name)
        try:
            if is_async_callable(fn):
                output = await asyncio.wait_for(
                    fn(**tool_call.arguments), timeout=effective_timeout
                )
            else:
                loop = asyncio.get_running_loop()
                output = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: fn(**tool_call.arguments)),
                    timeout=effective_timeout,
                )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - start) * 1000
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                is_error=True,
                error=f"Tool '{tool_call.name}' timed out after {effective_timeout}s",
                error_type="TimeoutError",
                duration_ms=duration_ms,
            )
            return result
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            error_msg = str(exc)
            error_type = type(exc).__name__
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=json.dumps({"error": error_msg, "type": error_type}),
                is_error=True,
                error=error_msg,
                error_type=error_type,
                duration_ms=duration_ms,
            )
            return result

        duration_ms = (time.monotonic() - start) * 1000
        content = self._serialize_output(output, tool_name=tool_call.name)
        truncated = content.endswith("[TRUNCATED]")

        result = ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=truncated,
            duration_ms=duration_ms,
        )

        if self._enable_dedup:
            self._dedup_cache[fingerprint] = result

        return result

    async def execute_parallel_async(
        self, tool_calls: list[CanonicalToolCall]
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel using asyncio.gather.

        Args:
            tool_calls: The list of canonical tool calls to execute.

        Returns:
            A list of ToolResult objects in input order.
        """
        if not tool_calls:
            return []

        tasks = [self.execute_async(tc) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))

    # ─── Internal helpers ───────────────────────────────────────────────

    def _resolve_timeout(self, tool_name: str) -> float:
        """Resolve the effective timeout for a tool name.

        Per-tool override takes precedence over the global per_tool_timeout.

        Args:
            tool_name: The name of the tool.

        Returns:
            The effective timeout in seconds.
        """
        return self._tool_timeouts.get(tool_name, self._per_tool_timeout)

    def _execute_with_timeout(
        self, fn: Callable[..., Any], arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Execute a callable with timeout enforcement via ThreadPoolExecutor.

        Args:
            fn: The callable to execute.
            arguments: The arguments to pass to the callable.
            timeout: Optional explicit timeout. Falls back to per_tool_timeout if None.

        Returns:
            The return value of the callable.

        Raises:
            FuturesTimeoutError: If execution exceeds the timeout.
        """
        effective_timeout = timeout if timeout is not None else self._per_tool_timeout
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(fn, **arguments)
            return future.result(timeout=effective_timeout)
        finally:
            # shutdown(wait=False) to avoid blocking on a timed-out thread
            executor.shutdown(wait=False)

    def _resolve_max_content_size(self, tool_name: str) -> int:
        """Resolve the effective max content size for a tool name.

        Per-tool override takes precedence over the global max_content_size.

        Args:
            tool_name: The name of the tool.

        Returns:
            The effective max content size in characters.
        """
        return self._max_content_sizes.get(tool_name, self._max_content_size)

    def _serialize_output(self, output: Any, tool_name: str | None = None) -> str:
        """Serialize tool output to a string with size enforcement.

        Attempts JSON serialization first; falls back to str() for
        non-JSON-serializable objects. If the resulting string exceeds
        the max content size for the tool, it is truncated with a
        [TRUNCATED] suffix.

        NOTE: This does NOT set is_error. Callers should check the return
        value for [TRUNCATED] suffix and set is_error=True if present.

        Args:
            output: The tool's return value.
            tool_name: Optional tool name for per-tool size override lookup.

        Returns:
            A string representation of the output, possibly truncated.
        """
        try:
            result = json.dumps(output)
        except (TypeError, ValueError):
            result = str(output)

        max_size = self._resolve_max_content_size(tool_name or "")
        if len(result) > max_size:
            result = result[:max_size] + "[TRUNCATED]"

        return result

    @staticmethod
    def _compute_fingerprint(name: str, arguments: dict[str, Any]) -> str:
        """Compute a SHA-256 fingerprint for a tool call.

        Uses json.dumps with sort_keys=True for deterministic hashing.

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
