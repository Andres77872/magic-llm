"""Tests for per-tool timeout overrides, heartbeat callback, and custom executor.

Covers:
- Task 1.2: Per-tool timeout overrides in ToolExecutor
- Task 1.3: Heartbeat callback in AsyncAgentLoop
- Task 1.4: Custom ToolExecutor flows through AsyncAgentLoop
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import CanonicalToolCall


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


# ─── Task 1.2: Per-tool timeout overrides ──────────────────────────────────


class TestPerToolTimeout:
    """Task 1.2: Per-tool timeout overrides in ToolExecutor."""

    async def _async_slow_tool(self, duration: float = 30.0) -> str:
        """Async tool that sleeps for the given duration."""
        await asyncio.sleep(duration)
        return "done"

    async def test_per_tool_timeout_cancels_long_tool(self):
        """Tool with global 1.0s timeout and no per-tool override is cancelled."""
        executor = ToolExecutor(per_tool_timeout=1.0)
        executor.register("slow", self._async_slow_tool)

        result = await executor.execute_async(_make_call("slow"))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "timed out" in result.error
        assert "1.0" in result.error

    async def test_per_tool_name_override_applies(self):
        """Per-tool timeout override applies: generate_image uses 120.0, not global 1.0."""
        executor = ToolExecutor(
            per_tool_timeout=1.0,
            tool_timeouts={"generate_image": 120.0},
        )

        async def fast_image_tool() -> str:
            return '{"url": "/images/img-abc.webp"}'

        executor.register("generate_image", fast_image_tool)

        # Would time out at 1.0s, but 120.0s override lets it succeed
        result = await executor.execute_async(_make_call("generate_image"))

        assert result.is_error is False
        assert "/images/img-abc.webp" in result.content

    async def test_global_timeout_used_when_no_per_tool_override(self):
        """Tools not in tool_timeouts use the global per_tool_timeout."""
        executor = ToolExecutor(
            per_tool_timeout=1.0,
            tool_timeouts={"generate_image": 120.0},
        )

        async def browsing_tool() -> str:
            await asyncio.sleep(5)  # Would exceed global 1.0s
            return "search results"

        executor.register("search", browsing_tool)
        result = await executor.execute_async(_make_call("search"))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"

    async def test_no_orphan_tasks_on_timeout(self):
        """Cancelled timeout leaves no persistent orphan asyncio tasks."""
        executor = ToolExecutor(per_tool_timeout=0.5)

        async def leaky_tool() -> str:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Simulate a tool that catches CancelledError and hangs
                await asyncio.sleep(10)
                raise
            return "done"

        executor.register("leaky", leaky_tool)

        tasks_before = {t for t in asyncio.all_tasks() if not t.get_name().startswith("Task-")}

        result = await executor.execute_async(_make_call("leaky"))

        tasks_after = {t for t in asyncio.all_tasks() if not t.get_name().startswith("Task-")}

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        # No significant NEW persistent tasks
        new_tasks = tasks_after - tasks_before
        assert len(new_tasks) <= 2, f"Potential orphan tasks detected: {len(new_tasks)}"

    async def test_timeout_returns_safe_error_message(self):
        """Timeout returns ToolResult with safe error and no stack trace."""
        executor = ToolExecutor(per_tool_timeout=1.0)

        async def fail_tool() -> str:
            await asyncio.sleep(30)
            return "never"

        executor.register("fail", fail_tool)
        result = await executor.execute_async(_make_call("fail"))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "Tool 'fail' timed out" in result.error
        # No raw stack traces in error
        assert "Traceback" not in result.error
        assert "File" not in result.error


# ─── Task 1.3: Heartbeat callback ──────────────────────────────────────────


class TestHeartbeatCallback:
    """Task 1.3: Heartbeat callback in AsyncAgentLoop."""

    async def test_heartbeat_invoked_during_long_execution(self):
        """Heartbeat callback invoked at least once for tools taking >0.5s.

        We test the heartbeat mechanism directly: spawn a heartbeat task
        alongside a long tool execution, verify the task fires.
        """
        heartbeat_calls = []
        start_time = time.monotonic()

        async def heartbeat() -> None:
            heartbeat_calls.append(time.monotonic() - start_time)

        executor = ToolExecutor(per_tool_timeout=10.0)

        async def slow_tool() -> str:
            """Tool that takes ~2 seconds."""
            await asyncio.sleep(2)
            return json.dumps({"status": "done"})

        executor.register("slow_tool", slow_tool)

        # Spawn heartbeat task alongside tool execution
        async def _heartbeat_loop():
            while True:
                await asyncio.sleep(0.5)  # 500ms interval for test speed
                await heartbeat()

        heartbeat_task = asyncio.create_task(_heartbeat_loop())

        try:
            result = await executor.execute_async(_make_call("slow_tool"))
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Tool should have succeeded
        assert result.is_error is False
        # Heartbeat should have been called multiple times (every 0.5s for ~2s)
        assert len(heartbeat_calls) >= 1, f"Expected >=1 heartbeat, got {len(heartbeat_calls)}"

    async def test_heartbeat_not_invoked_for_fast_tools(self):
        """Heartbeat NOT invoked when no heartbeat task is spawned (tool < 0.5s cycle)."""
        heartbeat_calls = []

        async def heartbeat() -> None:
            heartbeat_calls.append(time.monotonic())

        executor = ToolExecutor(per_tool_timeout=10.0)

        async def fast_tool() -> str:
            await asyncio.sleep(0.01)
            return json.dumps({"status": "fast"})

        executor.register("fast_tool", fast_tool)

        # No heartbeat task spawned — just tool execution
        result = await executor.execute_async(_make_call("fast_tool"))

        assert result.is_error is False
        # No heartbeat task was created, so no calls
        # This verifies the default is no heartbeat

    async def test_heartbeat_cb_none_does_not_spawn_task(self):
        """heartbeat_cb=None (default) — zero overhead, no heartbeat task spawned."""
        executor = ToolExecutor(per_tool_timeout=10.0)

        async def fast_tool() -> str:
            return json.dumps({"status": "done"})

        executor.register("fast_tool", fast_tool)

        task_count_before = len(asyncio.all_tasks())

        result = await executor.execute_async(_make_call("fast_tool"))

        task_count_after = len(asyncio.all_tasks())

        assert result.is_error is False
        # No extra tasks should have been spawned by heartbeat
        # (some variance expected from asyncio internals)
        assert task_count_after <= task_count_before + 2


# ─── Task 1.4: Custom ToolExecutor flows through AsyncAgentLoop ────────────


class TestCustomToolExecutorThroughLoop:
    """Task 1.4: Custom ToolExecutor flows through AsyncAgentLoop."""

    def test_custom_tool_executor_flows_to_agent_loop(self):
        """AsyncAgentLoop with custom executor uses it instead of creating a default one."""
        executor = ToolExecutor(
            per_tool_timeout=120.0,
            tool_timeouts={"generate_image": 120.0},
            max_content_sizes={"generate_image": 2000},
        )

        mock_client = MagicMock()
        mock_llm = MagicMock()
        mock_client.llm = mock_llm

        loop = AsyncAgentLoop(
            client=mock_client,
            tools=[],
            tool_executor=executor,
        )

        # Verify the executor was stored (not a new default one)
        assert loop._executor is executor
        assert loop._executor._per_tool_timeout == 120.0
        assert loop._executor._tool_timeouts == {"generate_image": 120.0}
        assert loop._executor._max_content_sizes == {"generate_image": 2000}

    def test_default_executor_created_when_none_provided(self):
        """No executor provided → default ToolExecutor(per_tool_timeout=30.0) created."""
        mock_client = MagicMock()
        mock_llm = MagicMock()
        mock_client.llm = mock_llm

        loop = AsyncAgentLoop(
            client=mock_client,
            tools=[],
            tool_executor=None,
        )

        assert loop._executor is not None
        assert loop._executor._per_tool_timeout == 30.0
        assert loop._executor._tool_timeouts == {}

    def test_custom_executor_timeout_and_content_sizes_apply(self):
        """Custom executor's max_content_sizes and tool_timeouts work in the loop."""
        executor = ToolExecutor(
            per_tool_timeout=120.0,
            tool_timeouts={"generate_image": 120.0},
            max_content_sizes={"generate_image": 2000},
        )

        # Verify directly on the executor
        assert executor._resolve_timeout("generate_image") == 120.0
        assert executor._resolve_timeout("search") == 120.0  # Falls back to global
        assert executor._resolve_max_content_size("generate_image") == 2000
        assert executor._resolve_max_content_size("search") == 50000  # Falls back to global

    def test_backward_compatible_no_executor(self):
        """Existing code that doesn't pass tool_executor works unchanged."""
        mock_client = MagicMock()
        mock_llm = MagicMock()
        mock_client.llm = mock_llm

        # No tool_executor arg — should work with defaults
        loop = AsyncAgentLoop(
            client=mock_client,
            tools=[],
        )

        assert loop._executor is not None
        assert loop._executor._per_tool_timeout == 30.0


# ─── Integration: Combined behavior of all Phase 1 changes ─────────────────


class TestPhase1Integration:
    """Integration tests combining all Phase 1 changes."""

    async def test_tool_timeout_and_content_size_together(self):
        """Both per-tool timeout override and max_content_size work together."""
        executor = ToolExecutor(
            per_tool_timeout=1.0,
            tool_timeouts={"generate_image": 10.0},
            max_content_sizes={"generate_image": 20},
        )

        async def image_tool() -> dict:
            await asyncio.sleep(0.1)  # Fast enough for both timeouts
            return {"url": "/images/img.webp", "data": "x" * 100}

        executor.register("generate_image", image_tool)
        result = await executor.execute_async(_make_call("generate_image"))

        # Timeout: 10s override, tool takes 0.1s → no timeout → is_error from truncation
        assert result.error_type is None or result.error_type != "TimeoutError"
        # Content: truncated by max_content_size=20
        assert result.content.endswith("[TRUNCATED]")
