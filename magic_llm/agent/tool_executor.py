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
import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.util import is_async_callable
from magic_llm.util.async_bridge import await_thread_future, submit_in_daemon_thread


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
        dedup_excluded_tools: set[str] | None = None,
        max_parallel_tools: int = 8,
        serial_tools: set[str] | None = None,
    ) -> None:
        if isinstance(max_parallel_tools, bool) or max_parallel_tools < 1:
            raise ValueError("max_parallel_tools must be a positive integer")
        self._max_parallel_tools = max_parallel_tools
        self._serial_tools = set(serial_tools or ())
        self._per_tool_timeout = per_tool_timeout
        self._enable_dedup = enable_dedup
        self._max_content_size = max_content_size
        self._max_content_sizes = max_content_sizes or {}
        self._tool_timeouts = tool_timeouts or {}
        self._dedup_excluded_tools = set(dedup_excluded_tools or set())
        self._registry: dict[str, Callable[..., Any]] = {}
        self._dedup_cache: dict[str, ToolResult] = {}

    def fork(self):
        """Return an isolated per-run registry and result cache."""
        clone = copy.copy(self)
        clone._registry = dict(self._registry)
        clone._dedup_cache = {}
        clone._dedup_excluded_tools = set(self._dedup_excluded_tools)
        clone._serial_tools = set(self._serial_tools)
        clone._max_content_sizes = dict(self._max_content_sizes)
        clone._tool_timeouts = dict(self._tool_timeouts)
        return clone

    def exclude_from_dedup(self, *names: str) -> None:
        """Exclude stateful tools from fingerprint deduplication."""
        self._dedup_excluded_tools.update(names)

    def serialize_tools(self, *names: str) -> None:
        """Make stateful calls ordered barriers within each model tool batch."""
        self._serial_tools.update(names)

    def _invalidate_tool_cache(self, name: str) -> None:
        self._dedup_cache = {key: value for key, value in self._dedup_cache.items()
                             if value.name != name}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a tool callable under the given name.

        Args:
            name: The tool name used for lookup during execution.
            fn: The callable to invoke when the tool is executed.
        """
        self._invalidate_tool_cache(name)
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
            self._invalidate_tool_cache(name)
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
        # Refuse calls whose arguments failed to parse upstream — running the
        # tool with silently-emptied arguments executes it with wrong input.
        malformed = self._malformed_arguments_result(tool_call)
        if malformed is not None:
            return malformed

        # Check dedup cache
        dedup_enabled = (
            self._enable_dedup and tool_call.name not in self._dedup_excluded_tools
        )
        if dedup_enabled:
            fingerprint = self._compute_fingerprint(
                tool_call.name, tool_call.arguments
            )
            if fingerprint in self._dedup_cache:
                cached = self._dedup_cache[fingerprint].model_copy()
                cached.tool_call_id = tool_call.id
                cached.is_deduplicated = True
                return cached

        start = time.monotonic()

        # Look up tool
        fn = self._registry.get(tool_call.name)
        if fn is None:
            return self._unknown_tool_result(tool_call, start)

        # Execute with timeout (per-tool override supported)
        effective_timeout = self._resolve_timeout(tool_call.name)
        future = submit_in_daemon_thread(fn, **tool_call.arguments)
        try:
            output = future.result(timeout=effective_timeout)
        except FuturesTimeoutError as exc:
            # On 3.11+ concurrent.futures.TimeoutError is the builtin
            # TimeoutError, so a TimeoutError raised *inside* the tool lands
            # here too. The future tells the two apart: an expired deadline
            # leaves it unfinished; a tool-raised TimeoutError completed it.
            if future.done():
                return self._error_result(tool_call, start, exc)
            return self._deadline_result(tool_call, start, effective_timeout)
        except Exception as exc:
            return self._error_result(tool_call, start, exc)

        duration_ms = (time.monotonic() - start) * 1000
        result = self._build_output_result(tool_call, output, duration_ms)

        # Cache for dedup
        if dedup_enabled and not result.is_error:
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

        results: list[ToolResult] = []
        for batch in self._ordered_batches(tool_calls):
            with ThreadPoolExecutor(max_workers=min(len(batch), self._max_parallel_tools)) as executor:
                results.extend(executor.map(self.execute, batch))
        return results

    def _ordered_batches(self, tool_calls: list[CanonicalToolCall]):
        # Stateful operations form barriers. Independent read tools may run
        # concurrently; repeated dedup fingerprints wait for their first result.
        batch: list[CanonicalToolCall] = []
        seen: set[str] = set()
        for call in tool_calls:
            fingerprint = self._compute_fingerprint(call.name, call.arguments)
            duplicate = (self._enable_dedup and call.name not in self._dedup_excluded_tools
                         and fingerprint in seen)
            if call.name in self._serial_tools or duplicate:
                if batch:
                    yield batch
                    batch = []
                yield [call]
            else:
                batch.append(call)
            seen.add(fingerprint)
        if batch:
            yield batch

    async def execute_async(self, tool_call: CanonicalToolCall) -> ToolResult:
        """Execute a single tool call, supporting both sync and async callables.

        If the callable is async, it is awaited directly. If sync, it runs
        in a worker thread to avoid blocking the event loop.

        Args:
            tool_call: The canonical tool call to execute.

        Returns:
            A ToolResult with the execution outcome.
        """
        malformed = self._malformed_arguments_result(tool_call)
        if malformed is not None:
            return malformed

        # Check dedup cache
        dedup_enabled = (
            self._enable_dedup and tool_call.name not in self._dedup_excluded_tools
        )
        if dedup_enabled:
            fingerprint = self._compute_fingerprint(
                tool_call.name, tool_call.arguments
            )
            if fingerprint in self._dedup_cache:
                cached = self._dedup_cache[fingerprint].model_copy()
                cached.tool_call_id = tool_call.id
                cached.is_deduplicated = True
                return cached

        start = time.monotonic()

        # Look up tool
        fn = self._registry.get(tool_call.name)
        if fn is None:
            return self._unknown_tool_result(tool_call, start)

        effective_timeout = self._resolve_timeout(tool_call.name)
        invocation: Any = None
        try:
            if is_async_callable(fn):
                invocation = asyncio.ensure_future(fn(**tool_call.arguments))
                output = await asyncio.wait_for(
                    invocation, timeout=effective_timeout
                )
            else:
                invocation = submit_in_daemon_thread(fn, **tool_call.arguments)
                output = await await_thread_future(
                    invocation, timeout=effective_timeout
                )
        except asyncio.TimeoutError as exc:
            # asyncio.TimeoutError is the builtin TimeoutError on 3.11+, so a
            # TimeoutError raised *by the tool itself* is caught here as well.
            # An expired executor deadline cancels the task (or leaves the
            # thread future unfinished); a tool-raised TimeoutError completes
            # the invocation normally.
            if isinstance(invocation, asyncio.Task):
                hit_deadline = invocation.cancelled() or not invocation.done()
            else:
                hit_deadline = invocation is None or not invocation.done()
            if not hit_deadline:
                return self._error_result(tool_call, start, exc)
            return self._deadline_result(tool_call, start, effective_timeout)
        except Exception as exc:
            return self._error_result(tool_call, start, exc)

        duration_ms = (time.monotonic() - start) * 1000
        result = self._build_output_result(tool_call, output, duration_ms)

        if dedup_enabled and not result.is_error:
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

        results: list[ToolResult] = []
        semaphore = asyncio.Semaphore(self._max_parallel_tools)

        async def execute_bounded(call: CanonicalToolCall) -> ToolResult:
            async with semaphore:
                return await self.execute_async(call)

        for batch in self._ordered_batches(tool_calls):
            tasks = [asyncio.create_task(execute_bounded(call)) for call in batch]
            try:
                results.extend(await asyncio.gather(*tasks))
            except BaseException:
                # gather does not cancel siblings when one child fails/cancels.
                # Never leave detached tools running after a batch has aborted.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        return results

    # ─── Internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _malformed_arguments_result(tool_call: CanonicalToolCall) -> ToolResult | None:
        """Return an error result for calls whose arguments failed to parse."""
        arguments_error = getattr(tool_call, "arguments_error", None)
        if not arguments_error:
            return None
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=json.dumps(
                {"error": arguments_error, "type": "MalformedArgumentsError"}
            ),
            is_error=True,
            error=arguments_error,
            error_type="MalformedArgumentsError",
            duration_ms=0.0,
        )

    @staticmethod
    def _unknown_tool_result(
        tool_call: CanonicalToolCall, start: float
    ) -> ToolResult:
        duration_ms = (time.monotonic() - start) * 1000
        error_msg = f"Unknown tool: {tool_call.name}"
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            # The envelope, not "": an empty tool message tells the model
            # nothing, and it happily narrates the call as having worked.
            content=json.dumps({"error": error_msg, "type": "UnknownToolError"}),
            is_error=True,
            error=error_msg,
            error_type="UnknownToolError",
            duration_ms=duration_ms,
        )

    @staticmethod
    def _deadline_result(
        tool_call: CanonicalToolCall, start: float, effective_timeout: float
    ) -> ToolResult:
        duration_ms = (time.monotonic() - start) * 1000
        error_msg = (
            f"Tool '{tool_call.name}' timed out after {effective_timeout}s"
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=json.dumps({"error": error_msg, "type": "TimeoutError"}),
            is_error=True,
            error=error_msg,
            error_type="TimeoutError",
            duration_ms=duration_ms,
        )

    @staticmethod
    def _error_result(
        tool_call: CanonicalToolCall, start: float, exc: BaseException
    ) -> ToolResult:
        duration_ms = (time.monotonic() - start) * 1000
        error_type = type(exc).__name__
        # str(exc) is "" for a bare raise of many exception types; an empty
        # error renders as an unexplained failure everywhere downstream.
        error_msg = str(exc) or f"Tool '{tool_call.name}' raised {error_type}"
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=json.dumps({"error": error_msg, "type": error_type}),
            is_error=True,
            error=error_msg,
            error_type=error_type,
            duration_ms=duration_ms,
        )

    def _build_output_result(
        self, tool_call: CanonicalToolCall, output: Any, duration_ms: float
    ) -> ToolResult:
        """Serialize a successful tool return, reporting truncation with a cause."""
        content, full_chars = self._serialize_output_with_size(
            output, tool_name=tool_call.name
        )
        limit = self._resolve_max_content_size(tool_call.name)
        truncated = full_chars > limit
        if truncated:
            error_msg = (
                f"Tool '{tool_call.name}' produced {full_chars} characters; "
                f"output was truncated to the {limit}-character limit"
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                is_error=True,
                error=error_msg,
                error_type="ContentTruncated",
                duration_ms=duration_ms,
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            is_error=False,
            duration_ms=duration_ms,
        )

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
        """Execute a callable in a daemon thread with timeout enforcement.

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
        future = submit_in_daemon_thread(fn, **arguments)
        return future.result(timeout=effective_timeout)

    def _resolve_max_content_size(self, tool_name: str) -> int:
        """Resolve the effective max content size for a tool name.

        Per-tool override takes precedence over the global max_content_size.

        Args:
            tool_name: The name of the tool.

        Returns:
            The effective max content size in characters.
        """
        return self._max_content_sizes.get(tool_name, self._max_content_size)

    def _serialize_output_with_size(
        self, output: Any, tool_name: str | None = None
    ) -> tuple[str, int]:
        """Serialize tool output, returning (content, pre-truncation length).

        Attempts JSON serialization first; falls back to str() for
        non-JSON-serializable objects. If the resulting string exceeds
        the max content size for the tool, it is truncated with a
        [TRUNCATED] suffix.
        """
        try:
            result = json.dumps(output)
        except (TypeError, ValueError):
            result = str(output)

        full_chars = len(result)
        max_size = self._resolve_max_content_size(tool_name or "")
        if full_chars > max_size:
            result = result[:max_size] + "[TRUNCATED]"

        return result, full_chars

    def _serialize_output(self, output: Any, tool_name: str | None = None) -> str:
        """Serialize tool output to a string with size enforcement.

        Retained for compatibility; see _serialize_output_with_size, which
        also reports the pre-truncation length so callers can attach a
        truncation cause instead of a bare error flag.
        """
        content, _ = self._serialize_output_with_size(output, tool_name=tool_name)
        return content

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
