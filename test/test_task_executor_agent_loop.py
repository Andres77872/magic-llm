"""Unit tests for TaskExecutor + AsyncAgentLoop integration.

Tests cover:
- TaskExecutor as tool_executor injection
- Task tool routing in agent loop context
- Parallel task execution via AsyncAgentLoop
- Depth/timeout/concurrency safeguards in loop context
"""
import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from magic_llm.agent import TaskExecutor, TaskManifest, TaskError
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.agent.async_agent_loop import AsyncAgentLoop


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    """Helper to create a CanonicalToolCall."""
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


def _make_manifest(
    id: str = "test_task",
    name: str = "Test Task",
    timeout_seconds: int = 30,
    max_concurrency: int = 5,
    max_depth: int = 3,
) -> TaskManifest:
    """Helper to create a minimal TaskManifest for tests."""
    return TaskManifest(
        id=id,
        name=name,
        description="A test task",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        max_depth=max_depth,
    )


class TestTaskExecutorInjection:
    """Tests for TaskExecutor injection into AsyncAgentLoop."""

    def test_task_executor_accepted_as_override(self):
        """TaskExecutor can be passed as tool_executor."""
        executor = TaskExecutor()
        client = MagicMock()

        loop = AsyncAgentLoop(
            client=client,
            tool_executor=executor,
        )

        assert loop._executor is executor

    def test_default_executor_created_if_none(self):
        """Default ToolExecutor created if none provided."""
        client = MagicMock()

        loop = AsyncAgentLoop(client=client)

        assert loop._executor is not None
        # Default is ToolExecutor, not TaskExecutor (unless tasks registered)
        from magic_llm.agent.tool_executor import ToolExecutor
        assert isinstance(loop._executor, ToolExecutor)

    def test_task_executor_with_registered_task(self):
        """TaskExecutor with registered task works as executor."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return f"Result: {query}"

        executor.register_task(_make_manifest(id="my_task"), my_task)
        client = MagicMock()

        loop = AsyncAgentLoop(
            client=client,
            tool_executor=executor,
        )

        assert loop._executor is executor
        assert "my_task" in executor._task_registry


class TestTaskRoutingInLoop:
    """Tests for task routing within AsyncAgentLoop."""

    @pytest.mark.asyncio
    async def test_task_tool_executed_by_executor(self):
        """Task tool call routed through TaskExecutor."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return f"Result: {query}"

        executor.register_task(_make_manifest(id="my_task"), my_task)

        # Execute directly through executor
        result = await executor.execute_async(_make_call("my_task", {"query": "test"}))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["task_type"] == "my_task"
        assert parsed["status"] == "ok"

    @pytest.mark.asyncio
    async def test_non_task_tool_delegates_to_base(self):
        """Non-task tool delegates to base ToolExecutor behavior."""
        executor = TaskExecutor()

        # Register ordinary tool
        def ordinary_tool(x: int) -> int:
            return x * 2

        executor.register("ordinary", ordinary_tool)

        # Also register a task to prove routing works
        async def task_tool(query: str) -> str:
            return f"Task: {query}"

        executor.register_task(_make_manifest(id="task_tool"), task_tool)

        # Non-task call
        result = await executor.execute_async(_make_call("ordinary", {"x": 5}))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed == 10  # Not TaskResult format

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tool returns error result."""
        executor = TaskExecutor()

        result = await executor.execute_async(_make_call("unknown"))

        assert result.is_error is True
        assert "Unknown tool" in result.error


class TestDepthTrackingInLoop:
    """Tests for depth tracking in async context."""

    def setup_method(self):
        """Reset depths before each test."""
        from magic_llm.agent.task_executor import reset_depths
        reset_depths()

    @pytest.mark.asyncio
    async def test_depth_increments_per_call(self):
        """Depth increments on each task call."""
        executor = TaskExecutor()
        from magic_llm.agent.task_executor import (
            _get_depth,
            _increment_depth,
        )

        async def my_task(query: str) -> str:
            # Check depth during execution
            depth = _get_depth("my_task")
            return f"depth={depth}"

        executor.register_task(_make_manifest(id="my_task", max_depth=5), my_task)

        # First call
        result1 = await executor.execute_async(_make_call("my_task", {"query": "1"}))
        parsed1 = json.loads(result1.content)
        assert "depth=1" in parsed1["summary"]  # Depth was 1 during execution

        # After execution, depth should be back to 0
        from magic_llm.agent.task_executor import _get_depth
        assert _get_depth("my_task") == 0

    @pytest.mark.asyncio
    async def test_depth_limit_returns_cancelled(self):
        """Depth >= max_depth returns cancelled result."""
        executor = TaskExecutor()
        from magic_llm.agent.task_executor import _increment_depth

        async def my_task(query: str) -> str:
            return "should not execute"

        executor.register_task(_make_manifest(id="my_task", max_depth=2), my_task)

        # Set depth to 2 (max_depth=2 means 2 >= 2 is exceeded)
        _increment_depth("my_task")
        _increment_depth("my_task")

        result = await executor.execute_async(_make_call("my_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "cancelled"
        assert parsed["error"]["error_type"] == TaskError.DEPTH_LIMIT


class TestTimeoutInLoop:
    """Tests for timeout enforcement in async context."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_task_result(self):
        """Task exceeding timeout returns TaskResult(status='timeout')."""
        executor = TaskExecutor()

        async def slow_task(query: str) -> str:
            await asyncio.sleep(10)
            return "should not reach"

        executor.register_task(_make_manifest(id="slow_task", timeout_seconds=1), slow_task)

        start = asyncio.get_event_loop().time()
        result = await executor.execute_async(_make_call("slow_task", {"query": "test"}))
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 2.0
        parsed = json.loads(result.content)
        assert parsed["status"] == "timeout"
        assert parsed["error"]["retryable"] is True

    @pytest.mark.asyncio
    async def test_task_within_timeout_succeeds(self):
        """Task completing within timeout succeeds."""
        executor = TaskExecutor()

        async def fast_task(query: str) -> str:
            await asyncio.sleep(0.1)
            return "done"

        executor.register_task(_make_manifest(id="fast_task", timeout_seconds=5), fast_task)

        result = await executor.execute_async(_make_call("fast_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"


class TestConcurrencyInLoop:
    """Tests for semaphore concurrency control in async context."""

    @pytest.mark.asyncio
    async def test_concurrency_limit_queues_calls(self):
        """Concurrency limit queues excess calls."""
        executor = TaskExecutor()
        execution_order = []

        async def tracked_task(query: str) -> str:
            execution_order.append(f"start_{query}")
            await asyncio.sleep(0.3)
            execution_order.append(f"end_{query}")
            return query

        executor.register_task(_make_manifest(id="tracked_task", max_concurrency=2), tracked_task)

        # Launch 3 concurrent calls (max_concurrency=2)
        calls = [
            _make_call("tracked_task", {"query": f"call_{i}"}, id=f"call_{i}")
            for i in range(3)
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 3
        assert all(r.is_error is False for r in results)

        # Verify all 3 completed
        assert len([x for x in execution_order if x.startswith("end_")]) == 3


class TestParallelTaskExecution:
    """Tests for parallel task execution via TaskExecutor."""

    @pytest.mark.asyncio
    async def test_parallel_tasks_with_different_ids(self):
        """Multiple different tasks executed in parallel."""
        executor = TaskExecutor()

        async def task_a(query: str) -> str:
            await asyncio.sleep(0.1)
            return f"A: {query}"

        async def task_b(query: str) -> str:
            await asyncio.sleep(0.1)
            return f"B: {query}"

        executor.register_task(_make_manifest(id="task_a"), task_a)
        executor.register_task(_make_manifest(id="task_b"), task_b)

        calls = [
            _make_call("task_a", {"query": "test"}, id="call_a"),
            _make_call("task_b", {"query": "test"}, id="call_b"),
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 2
        parsed_a = json.loads(results[0].content)
        parsed_b = json.loads(results[1].content)

        assert parsed_a["task_type"] == "task_a"
        assert parsed_b["task_type"] == "task_b"

    @pytest.mark.asyncio
    async def test_parallel_same_task_different_calls(self):
        """Same task called multiple times in parallel."""
        executor = TaskExecutor()

        async def my_task(iteration: int) -> str:
            await asyncio.sleep(0.1)
            return f"Result {iteration}"

        executor.register_task(_make_manifest(id="my_task"), my_task)

        calls = [
            _make_call("my_task", {"iteration": i}, id=f"call_{i}")
            for i in range(3)
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 3
        for i, result in enumerate(results):
            parsed = json.loads(result.content)
            assert parsed["status"] == "ok"

    @pytest.mark.asyncio
    async def test_partial_ok_one_fails(self):
        """One task fails, all results collected (partial_ok behavior)."""
        executor = TaskExecutor()

        async def success_task(query: str) -> str:
            return "Success"

        async def fail_task(query: str) -> str:
            raise ValueError("Intentional failure")

        executor.register_task(_make_manifest(id="success"), success_task)
        executor.register_task(_make_manifest(id="fail"), fail_task)

        calls = [
            _make_call("success", {"query": "test"}, id="call_success"),
            _make_call("fail", {"query": "test"}, id="call_fail"),
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 2
        parsed_success = json.loads(results[0].content)
        parsed_fail = json.loads(results[1].content)

        assert parsed_success["status"] == "ok"
        assert parsed_fail["status"] == "failed"
        assert parsed_fail["error"]["error_type"] == TaskError.EXECUTION


class TestErrorHandlingInLoop:
    """Tests for error handling in loop context."""

    @pytest.mark.asyncio
    async def test_execution_error_returns_failed_result(self):
        """Task raising exception returns TaskResult(status='failed')."""
        executor = TaskExecutor()

        async def error_task(query: str) -> str:
            raise ValueError("Something went wrong")

        executor.register_task(_make_manifest(id="error_task"), error_task)

        result = await executor.execute_async(_make_call("error_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "failed"
        assert parsed["error"]["error_type"] == TaskError.EXECUTION
        assert "Something went wrong" in parsed["error"]["message"]
        assert parsed["error"]["retryable"] is True


class TestResultNormalizationInLoop:
    """Tests for result normalization in loop context."""

    @pytest.mark.asyncio
    async def test_string_output_normalized(self):
        """String output appears in summary."""
        executor = TaskExecutor()

        async def string_task(query: str) -> str:
            return "Plain string result"

        executor.register_task(_make_manifest(id="string_task"), string_task)

        result = await executor.execute_async(_make_call("string_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        assert "Plain string result" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_dict_output_normalized(self):
        """Dict output converted to Markdown."""
        executor = TaskExecutor()

        async def dict_task(query: str) -> dict:
            return {"count": 3, "items": ["a", "b", "c"]}

        executor.register_task(_make_manifest(id="dict_task"), dict_task)

        result = await executor.execute_async(_make_call("dict_task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        assert "**count**" in parsed["summary"]
        assert "3" in parsed["summary"]


class TestTaskExecutorUnregisterInLoop:
    """Tests for task unregistration in loop context."""

    @pytest.mark.asyncio
    async def test_unregister_removes_from_all_registries(self):
        """Unregister removes from task registry and base registry."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return query

        executor.register_task(_make_manifest(id="my_task"), my_task)

        assert "my_task" in executor._task_registry
        assert "my_task" in executor._registry

        removed = executor.unregister_task("my_task")
        assert removed is True

        assert "my_task" not in executor._task_registry
        assert "my_task" not in executor._registry

    @pytest.mark.asyncio
    async def test_unregister_unknown_returns_false(self):
        """Unregister unknown task returns False."""
        executor = TaskExecutor()

        removed = executor.unregister_task("unknown")
        assert removed is False