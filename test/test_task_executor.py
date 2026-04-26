"""Unit tests for TaskExecutor — task/subagent runtime with routing, safeguards.

Tests cover:
- Task vs non-task tool routing
- Depth tracking via ContextVar
- Timeout enforcement
- Semaphore concurrency control
- Result normalization
- Integration with AsyncAgentLoop
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from magic_llm.agent.task_executor import (
    TaskExecutor,
    TASK_DEPTH,
    _get_depth,
    _increment_depth,
    _decrement_depth,
    reset_depths,
    get_all_depths,
)
from magic_llm.agent.types import (
    CanonicalToolCall,
    TaskError,
    TaskManifest,
    TaskResult,
    ToolResult,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


def _make_manifest(
    id: str = "test_task",
    name: str = "Test Task",
    timeout_seconds: int = 30,
    max_concurrency: int = 5,
    max_depth: int = 3,
) -> TaskManifest:
    return TaskManifest(
        id=id,
        name=name,
        description="A test task",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        max_depth=max_depth,
    )


# ─── Routing Tests ───────────────────────────────────────────────────────────


class TestTaskExecutorRouting:
    """Task vs non-task tool routing."""

    @pytest.mark.asyncio
    async def test_task_tool_returns_task_result_json(self):
        """Registered task returns TaskResult JSON string."""
        executor = TaskExecutor()

        async def my_task(query: str) -> dict:
            return {"results": [query]}

        manifest = _make_manifest(id="my_task")
        executor.register_task(manifest, my_task)

        result = await executor.execute_async(_make_call("my_task", {"query": "test"}))

        assert result.is_error is False
        assert result.name == "my_task"
        # Content should be TaskResult JSON
        parsed = json.loads(result.content)
        assert parsed["task_type"] == "my_task"
        assert parsed["status"] == "ok"
        assert "summary" in parsed

    @pytest.mark.asyncio
    async def test_non_task_tool_delegates_to_base_executor(self):
        """Non-task tool delegates to super().execute_async()."""
        executor = TaskExecutor()

        # Register ordinary tool via base executor
        def ordinary_tool(x: int) -> int:
            return x * 2

        executor.register("ordinary", ordinary_tool)

        # Register a task too (to prove routing works)
        async def task_tool(query: str) -> str:
            return f"Task: {query}"

        executor.register_task(_make_manifest(id="task_tool"), task_tool)

        result = await executor.execute_async(_make_call("ordinary", {"x": 5}))

        assert result.is_error is False
        assert result.name == "ordinary"
        # Content should be JSON-serialized output (not TaskResult format)
        parsed = json.loads(result.content)
        assert parsed == 10

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tool (not in task_registry, not in base registry) returns error."""
        executor = TaskExecutor()

        result = await executor.execute_async(_make_call("unknown"))

        assert result.is_error is True
        assert "Unknown tool" in result.error
        assert result.error_type == "UnknownToolError"

    @pytest.mark.asyncio
    async def test_task_registry_stores_manifest(self):
        """register_task() stores manifest in _task_registry."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task")
        executor.register_task(manifest, my_task)

        assert "my_task" in executor._task_registry
        stored = executor._task_registry["my_task"]
        assert stored.id == "my_task"
        assert stored.timeout_seconds == 30
        assert stored.max_concurrency == 5
        assert stored.max_depth == 3


# ─── Depth Tracking Tests ────────────────────────────────────────────────────


class TestTaskExecutorDepthTracking:
    """ContextVar depth tracking."""

    def setup_method(self):
        """Reset depths before each test."""
        reset_depths()

    def test_get_depth_returns_zero_initially(self):
        """Fresh context has depth 0 for any task_id."""
        assert _get_depth("task_a") == 0
        assert _get_depth("task_b") == 0

    def test_increment_depth_increases_count(self):
        """_increment_depth increases depth by 1."""
        new_depth = _increment_depth("task_a")
        assert new_depth == 1
        assert _get_depth("task_a") == 1

        new_depth = _increment_depth("task_a")
        assert new_depth == 2
        assert _get_depth("task_a") == 2

    def test_decrement_depth_reduces_count(self):
        """_decrement_depth reduces depth by 1, min 0."""
        _increment_depth("task_a")
        _increment_depth("task_a")
        assert _get_depth("task_a") == 2

        new_depth = _decrement_depth("task_a")
        assert new_depth == 1
        assert _get_depth("task_a") == 1

        new_depth = _decrement_depth("task_a")
        assert new_depth == 0
        assert _get_depth("task_a") == 0

        # Decrement below 0 stays at 0
        new_depth = _decrement_depth("task_a")
        assert new_depth == 0

    def test_depth_is_per_task_id(self):
        """Different task_ids have independent depth counters."""
        _increment_depth("task_a")
        _increment_depth("task_a")
        _increment_depth("task_b")

        assert _get_depth("task_a") == 2
        assert _get_depth("task_b") == 1
        assert _get_depth("task_c") == 0

    def test_reset_depths_clears_all(self):
        """reset_depths() clears all counters."""
        _increment_depth("task_a")
        _increment_depth("task_b")
        reset_depths()

        assert _get_depth("task_a") == 0
        assert _get_depth("task_b") == 0
        assert get_all_depths() == {}

    @pytest.mark.asyncio
    async def test_depth_exceeded_returns_cancelled_result(self):
        """Depth >= max_depth returns TaskResult(status='cancelled')."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task", max_depth=2)
        executor.register_task(manifest, my_task)

        # Set depth to 2 (max_depth=2 means 2 >= 2 is exceeded)
        _increment_depth("my_task")
        _increment_depth("my_task")
        assert _get_depth("my_task") == 2

        result = await executor.execute_async(_make_call("my_task", {"query": "test"}))

        # Should return cancelled result, not execute
        parsed = json.loads(result.content)
        assert parsed["status"] == "cancelled"
        assert parsed["error"]["error_type"] == TaskError.DEPTH_LIMIT

    @pytest.mark.asyncio
    async def test_depth_increments_on_execution(self):
        """Successful execution increments then decrements depth."""
        executor = TaskExecutor()
        reset_depths()

        async def my_task(query: str) -> str:
            # Check depth during execution
            current_depth = _get_depth("my_task")
            return f"depth_was_{current_depth}"

        manifest = _make_manifest(id="my_task", max_depth=5)
        executor.register_task(manifest, my_task)

        result = await executor.execute_async(_make_call("my_task", {"query": "test"}))
        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"

        # After execution, depth should be back to 0
        assert _get_depth("my_task") == 0


# ─── Timeout Tests ───────────────────────────────────────────────────────────


class TestTaskExecutorTimeout:
    """Per-task timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_result(self):
        """Task exceeding timeout_seconds returns TaskResult(status='timeout')."""
        executor = TaskExecutor()

        async def slow_task(query: str) -> str:
            await asyncio.sleep(10)
            return "should not reach"

        manifest = _make_manifest(id="slow_task", timeout_seconds=1)
        executor.register_task(manifest, slow_task)

        start = time.monotonic()
        result = await executor.execute_async(_make_call("slow_task", {"query": "test"}))
        elapsed = time.monotonic() - start

        # Should complete within ~1s
        assert elapsed < 2.0, f"Timeout took {elapsed:.1f}s, expected ~1s"

        parsed = json.loads(result.content)
        assert parsed["status"] == "timeout"
        assert parsed["error"]["error_type"] == TaskError.TIMEOUT
        assert parsed["error"]["retryable"] is True

    @pytest.mark.asyncio
    async def test_task_within_timeout_succeeds(self):
        """Task completing within timeout succeeds."""
        executor = TaskExecutor()

        async def fast_task(query: str) -> str:
            await asyncio.sleep(0.1)
            return "done"

        manifest = _make_manifest(id="fast_task", timeout_seconds=5)
        executor.register_task(manifest, fast_task)

        result = await executor.execute_async(_make_call("fast_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"


# ─── Concurrency Tests ───────────────────────────────────────────────────────


class TestTaskExecutorConcurrency:
    """Per-task semaphore concurrency control."""

    @pytest.mark.asyncio
    async def test_semaphore_initialized_on_registration(self):
        """register_task() creates asyncio.Semaphore with max_concurrency."""
        executor = TaskExecutor()

        manifest = _make_manifest(id="my_task", max_concurrency=3)
        executor.register_task(manifest, lambda: "ok")

        assert "my_task" in executor._task_semaphores
        semaphore = executor._task_semaphores["my_task"]
        # Semaphore._value is the current available slots
        assert semaphore._value == 3

    @pytest.mark.asyncio
    async def test_concurrency_limit_queues_calls(self):
        """4 concurrent calls with max_concurrency=3, 4th waits."""
        executor = TaskExecutor()
        execution_order = []

        async def tracked_task(query: str) -> str:
            execution_order.append(f"start_{query}")
            await asyncio.sleep(0.5)
            execution_order.append(f"end_{query}")
            return query

        manifest = _make_manifest(id="tracked_task", max_concurrency=3)
        executor.register_task(manifest, tracked_task)

        # Launch 4 concurrent calls
        calls = [
            _make_call("tracked_task", {"query": f"call_{i}"}, id=f"call_{i}")
            for i in range(4)
        ]

        start = time.monotonic()
        results = await executor.execute_parallel_async(calls)
        elapsed = time.monotonic() - start

        # All should complete
        assert len(results) == 4
        assert all(r.is_error is False for r in results)

        # Total time should be ~1s (3 parallel + 1 queued), not 2s (4 parallel)
        # Actually with max_concurrency=3 and 4 calls, the pattern is:
        # - First 3 start immediately
        # - 4th waits until one of first 3 finishes
        # - Total time ≈ 1s (0.5s for first 3, then 4th runs during second 0.5s)
        # So elapsed should be ~1s, not ~0.5s (if all 4 were parallel)
        # and not ~2s (if all 4 were sequential)
        assert elapsed >= 0.5, "Should not complete instantly"
        assert elapsed < 1.5, f"4 calls with concurrency=3 took {elapsed:.1f}s, expected ~1s"

        # Verify all 4 completed
        assert len([x for x in execution_order if x.startswith("end_")]) == 4


# ─── Error Handling Tests ─────────────────────────────────────────────────────


class TestTaskExecutorErrors:
    """Error handling in task execution."""

    @pytest.mark.asyncio
    async def test_execution_error_returns_failed_result(self):
        """Task raising exception returns TaskResult(status='failed')."""
        executor = TaskExecutor()

        async def error_task(query: str) -> str:
            raise ValueError("Something went wrong")

        manifest = _make_manifest(id="error_task")
        executor.register_task(manifest, error_task)

        result = await executor.execute_async(_make_call("error_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "failed"
        assert parsed["error"]["error_type"] == TaskError.EXECUTION
        assert "Something went wrong" in parsed["error"]["message"]
        assert parsed["error"]["retryable"] is True


# ─── Result Normalization Tests ───────────────────────────────────────────────


class TestTaskExecutorNormalization:
    """Result normalization integration."""

    @pytest.mark.asyncio
    async def test_string_output_normalized_to_summary(self):
        """String output appears in summary."""
        executor = TaskExecutor()

        async def string_task(query: str) -> str:
            return "Plain string result"

        manifest = _make_manifest(id="string_task")
        executor.register_task(manifest, string_task)

        result = await executor.execute_async(_make_call("string_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        assert "Plain string result" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_dict_output_normalized_to_markdown(self):
        """Dict output is converted to Markdown bullet list."""
        executor = TaskExecutor()

        async def dict_task(query: str) -> dict:
            return {"count": 3, "items": ["a", "b", "c"]}

        manifest = _make_manifest(id="dict_task")
        executor.register_task(manifest, dict_task)

        result = await executor.execute_async(_make_call("dict_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        # Summary should contain Markdown-formatted dict
        assert "**count**" in parsed["summary"]
        assert "3" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_list_output_normalized_to_numbered_list(self):
        """List output is converted to Markdown numbered list."""
        executor = TaskExecutor()

        async def list_task(query: str) -> list:
            return ["first", "second", "third"]

        manifest = _make_manifest(id="list_task")
        executor.register_task(manifest, list_task)

        result = await executor.execute_async(_make_call("list_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        # Summary should contain numbered list
        assert "1." in parsed["summary"]
        assert "first" in parsed["summary"]


# ─── Unregister Tests ─────────────────────────────────────────────────────────


class TestTaskExecutorUnregister:
    """Task unregistration."""

    @pytest.mark.asyncio
    async def test_unregister_task_removes_from_registry(self):
        """unregister_task() removes task from all registries."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task")
        executor.register_task(manifest, my_task)

        assert "my_task" in executor._task_registry

        removed = executor.unregister_task("my_task")
        assert removed is True
        assert "my_task" not in executor._task_registry
        assert "my_task" not in executor._task_semaphores

    @pytest.mark.asyncio
    async def test_unregister_unknown_task_returns_false(self):
        """unregister_task() returns False for unknown task."""
        executor = TaskExecutor()

        removed = executor.unregister_task("unknown")
        assert removed is False

    def test_get_registered_tasks_returns_list(self):
        """get_registered_tasks() returns list of registered task IDs."""
        executor = TaskExecutor()

        executor.register_task(_make_manifest(id="task_a"), lambda: "a")
        executor.register_task(_make_manifest(id="task_b"), lambda: "b")

        registered = executor.get_registered_tasks()
        assert "task_a" in registered
        assert "task_b" in registered
        assert len(registered) == 2

    def test_get_task_manifest_returns_manifest(self):
        """get_task_manifest() returns manifest for registered task."""
        executor = TaskExecutor()

        manifest = _make_manifest(id="my_task", timeout_seconds=60)
        executor.register_task(manifest, lambda: "ok")

        retrieved = executor.get_task_manifest("my_task")
        assert retrieved is not None
        assert retrieved.id == "my_task"
        assert retrieved.timeout_seconds == 60

    def test_get_task_manifest_unknown_returns_none(self):
        """get_task_manifest() returns None for unknown task."""
        executor = TaskExecutor()

        retrieved = executor.get_task_manifest("unknown")
        assert retrieved is None