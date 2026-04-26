"""E2E tests for task runtime execution.

Tests cover real execution scenarios that were previously only tested in magic-agents.
These tests verify the complete task execution path from registration to result
without relying on magic-agents wrapper infrastructure.
"""
import asyncio
import json
import pytest
import uuid

from magic_llm.agent import TaskExecutor, TaskManifest, TaskError, ResultNormalizer
from magic_llm.agent.types import CanonicalToolCall


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    """Helper to create a CanonicalToolCall."""
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


def _make_manifest(
    id: str = "test_task",
    name: str = "Test Task",
    description: str = "A test task",
    timeout_seconds: int = 30,
    max_concurrency: int = 5,
    max_depth: int = 3,
    input_schema: dict = None,
) -> TaskManifest:
    """Helper to create a minimal TaskManifest for tests."""
    return TaskManifest(
        id=id,
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {"query": {"type": "string"}}},
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        max_depth=max_depth,
    )


class TestE2ESingleTaskExecution:
    """E2E tests for single task execution."""

    @pytest.mark.asyncio
    async def test_task_returns_task_result_json(self):
        """Task execution returns TaskResult JSON string."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return f"Processed: {query}"

        executor.register_task(_make_manifest(id="my_task"), my_task)

        result = await executor.execute_async(_make_call("my_task", {"query": "hello"}))

        assert result.is_error is False
        parsed = json.loads(result.content)

        assert parsed["task_id"] is not None
        assert parsed["task_type"] == "my_task"
        assert parsed["status"] == "ok"
        assert "Processed: hello" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_task_with_complex_output(self):
        """Task with complex dict output normalized correctly."""
        executor = TaskExecutor()

        async def research_task(query: str) -> dict:
            return {
                "sources": ["source1", "source2", "source3"],
                "summary": f"Key findings for {query}",
                "confidence": 0.85,
            }

        executor.register_task(_make_manifest(id="research"), research_task)

        result = await executor.execute_async(_make_call("research", {"query": "AI trends"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        assert "**sources**" in parsed["summary"]
        assert "**summary**" in parsed["summary"]
        assert "0.85" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_task_with_markdown_output(self):
        """Task returning Markdown preserves formatting."""
        executor = TaskExecutor()

        async def markdown_task(query: str) -> str:
            return """# Research Results

## Sources
1. Source A
2. Source B

## Summary
Key findings from research.
"""

        executor.register_task(_make_manifest(id="markdown"), markdown_task)

        result = await executor.execute_async(_make_call("markdown", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"
        assert "# Research Results" in parsed["summary"]
        assert "## Sources" in parsed["summary"]


class TestE2EParallelExecution:
    """E2E tests for parallel task execution."""

    @pytest.mark.asyncio
    async def test_parallel_invocations_complete(self):
        """Multiple parallel invocations all complete."""
        executor = TaskExecutor()

        async def agent_a(query: str) -> str:
            await asyncio.sleep(0.1)
            return f"Agent A: {query}"

        async def agent_b(query: str) -> str:
            await asyncio.sleep(0.1)
            return f"Agent B: {query}"

        executor.register_task(_make_manifest(id="agent.a"), agent_a)
        executor.register_task(_make_manifest(id="agent.b"), agent_b)

        calls = [
            _make_call("agent.a", {"query": "test"}, id="call_a"),
            _make_call("agent.b", {"query": "test"}, id="call_b"),
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 2
        for result in results:
            parsed = json.loads(result.content)
            assert parsed["status"] == "ok"

    @pytest.mark.asyncio
    async def test_partial_ok_one_fails(self):
        """One task fails, all results collected (partial_ok)."""
        executor = TaskExecutor()

        async def success_func(query: str) -> str:
            return "Success result"

        async def fail_func(query: str) -> str:
            raise ValueError("Intentional failure")

        executor.register_task(_make_manifest(id="success.agent"), success_func)
        executor.register_task(_make_manifest(id="fail.agent"), fail_func)

        calls = [
            _make_call("success.agent", {"query": "test"}, id="call_success"),
            _make_call("fail.agent", {"query": "test"}, id="call_fail"),
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 2

        parsed_success = json.loads(results[0].content)
        parsed_fail = json.loads(results[1].content)

        assert parsed_success["status"] == "ok"
        assert parsed_fail["status"] == "failed"
        assert parsed_fail["error"]["error_type"] == TaskError.EXECUTION

    @pytest.mark.asyncio
    async def test_three_parallel_all_complete(self):
        """Three parallel invocations all complete."""
        executor = TaskExecutor()

        async def parallel_task(iteration: int) -> str:
            await asyncio.sleep(0.05)
            return f"Result {iteration}"

        executor.register_task(_make_manifest(id="parallel.test"), parallel_task)

        calls = [
            _make_call("parallel.test", {"iteration": 1}, id="call_1"),
            _make_call("parallel.test", {"iteration": 2}, id="call_2"),
            _make_call("parallel.test", {"iteration": 3}, id="call_3"),
        ]

        results = await executor.execute_parallel_async(calls)

        assert len(results) == 3
        for i, result in enumerate(results, 1):
            parsed = json.loads(result.content)
            assert parsed["status"] == "ok"
            assert f"Result {i}" in parsed["summary"]


class TestE2ETimeoutHandling:
    """E2E tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_one_task_timeout_others_complete(self):
        """One task times out, others complete."""
        executor = TaskExecutor()

        async def quick_func(query: str) -> str:
            await asyncio.sleep(0.1)
            return "Quick result"

        async def slow_func(query: str) -> str:
            await asyncio.sleep(5)  # Exceeds timeout
            return "Should not reach"

        executor.register_task(_make_manifest(id="quick.agent", timeout_seconds=10), quick_func)
        executor.register_task(_make_manifest(id="slow.agent", timeout_seconds=1), slow_func)

        calls = [
            _make_call("quick.agent", {"query": "test"}, id="call_quick"),
            _make_call("slow.agent", {"query": "test"}, id="call_slow"),
        ]

        results = await executor.execute_parallel_async(calls)

        parsed_quick = json.loads(results[0].content)
        parsed_slow = json.loads(results[1].content)

        assert parsed_quick["status"] == "ok"
        assert parsed_slow["status"] == "timeout"
        assert parsed_slow["error"]["error_type"] == TaskError.TIMEOUT

    @pytest.mark.asyncio
    async def test_task_within_custom_timeout_succeeds(self):
        """Task completing within custom timeout succeeds."""
        executor = TaskExecutor()

        async def medium_task(query: str) -> str:
            await asyncio.sleep(2)
            return "Medium result"

        executor.register_task(_make_manifest(id="medium.task", timeout_seconds=5), medium_task)

        result = await executor.execute_async(_make_call("medium.task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"


class TestE2EDepthLimit:
    """E2E tests for depth limit enforcement."""

    def setup_method(self):
        """Reset depths before each test."""
        from magic_llm.agent.task_executor import reset_depths
        reset_depths()

    @pytest.mark.asyncio
    async def test_depth_limit_exceeded(self):
        """Depth limit exceeded returns cancelled result."""
        executor = TaskExecutor()
        from magic_llm.agent.task_executor import _increment_depth

        async def nested_task(query: str) -> str:
            return "should not execute"

        executor.register_task(_make_manifest(id="nested.task", max_depth=1), nested_task)

        # Simulate depth already at max
        _increment_depth("nested.task")

        result = await executor.execute_async(_make_call("nested.task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "cancelled"
        assert parsed["error"]["error_type"] == TaskError.DEPTH_LIMIT


class TestE2EConcurrencyLimit:
    """E2E tests for concurrency limit enforcement."""

    @pytest.mark.asyncio
    async def test_concurrency_queues_excess_calls(self):
        """Concurrency limit queues excess calls."""
        executor = TaskExecutor()
        execution_times = []

        async def concurrent_task(query: str) -> str:
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.2)
            end = asyncio.get_event_loop().time()
            execution_times.append((start, end))
            return f"Done: {query}"

        executor.register_task(_make_manifest(id="concurrent.task", max_concurrency=2), concurrent_task)

        # Launch 4 calls with max_concurrency=2
        calls = [
            _make_call("concurrent.task", {"query": f"call_{i}"}, id=f"call_{i}")
            for i in range(4)
        ]

        start_time = asyncio.get_event_loop().time()
        results = await executor.execute_parallel_async(calls)
        elapsed = asyncio.get_event_loop().time() - start_time

        assert len(results) == 4
        assert all(r.is_error is False for r in results)

        # With max_concurrency=2 and 4 calls, elapsed should be ~0.4s
        # (2 parallel batches, each ~0.2s)
        assert elapsed >= 0.3  # Not instantaneous
        assert elapsed < 1.0  # Not fully sequential


class TestE2EErrorHandling:
    """E2E tests for error handling."""

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Task with input validation error handled."""
        executor = TaskExecutor()

        # Task that validates input
        async def validating_task(query: str) -> str:
            if not query:
                raise ValueError("Query cannot be empty")
            return f"Processed: {query}"

        executor.register_task(_make_manifest(id="validating"), validating_task)

        # Valid call
        result_valid = await executor.execute_async(_make_call("validating", {"query": "test"}))
        parsed_valid = json.loads(result_valid.content)
        assert parsed_valid["status"] == "ok"

        # Invalid call (empty query triggers validation error)
        result_invalid = await executor.execute_async(_make_call("validating", {"query": ""}))
        parsed_invalid = json.loads(result_invalid.content)
        assert parsed_invalid["status"] == "failed"
        assert "Query cannot be empty" in parsed_invalid["error"]["message"]

    @pytest.mark.asyncio
    async def test_runtime_error_handling(self):
        """Task with runtime error handled."""
        executor = TaskExecutor()

        async def error_task(query: str) -> str:
            raise RuntimeError("Internal error")

        executor.register_task(_make_manifest(id="error.task"), error_task)

        result = await executor.execute_async(_make_call("error.task", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["status"] == "failed"
        assert parsed["error"]["error_type"] == TaskError.EXECUTION
        assert "Internal error" in parsed["error"]["message"]


class TestE2ENormalization:
    """E2E tests for result normalization."""

    @pytest.mark.asyncio
    async def test_string_normalization(self):
        """String output normalized directly."""
        executor = TaskExecutor()

        async def string_task(query: str) -> str:
            return "Plain text result"

        executor.register_task(_make_manifest(id="string"), string_task)

        result = await executor.execute_async(_make_call("string", {"query": "test"}))

        parsed = json.loads(result.content)
        assert parsed["summary"] == "Plain text result"

    @pytest.mark.asyncio
    async def test_dict_normalization(self):
        """Dict output normalized to Markdown."""
        executor = TaskExecutor()

        async def dict_task(query: str) -> dict:
            return {"count": 5, "items": ["a", "b", "c"], "status": "complete"}

        executor.register_task(_make_manifest(id="dict"), dict_task)

        result = await executor.execute_async(_make_call("dict", {"query": "test"}))

        parsed = json.loads(result.content)
        assert "**count**" in parsed["summary"]
        assert "**items**" in parsed["summary"]
        assert "**status**" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_list_normalization(self):
        """List output normalized to numbered list."""
        executor = TaskExecutor()

        async def list_task(query: str) -> list:
            return ["Item 1", "Item 2", "Item 3"]

        executor.register_task(_make_manifest(id="list"), list_task)

        result = await executor.execute_async(_make_call("list", {"query": "test"}))

        parsed = json.loads(result.content)
        assert "1. Item 1" in parsed["summary"]
        assert "2. Item 2" in parsed["summary"]
        assert "3. Item 3" in parsed["summary"]


class TestE2ETaskIdGeneration:
    """E2E tests for task ID generation."""

    @pytest.mark.asyncio
    async def test_unique_task_id_per_invocation(self):
        """Each invocation gets unique task_id."""
        executor = TaskExecutor()

        async def my_task(query: str) -> str:
            return "result"

        executor.register_task(_make_manifest(id="my_task"), my_task)

        results = await asyncio.gather(
            executor.execute_async(_make_call("my_task", {"query": "1"})),
            executor.execute_async(_make_call("my_task", {"query": "2"})),
        )

        ids = [json.loads(r.content)["task_id"] for r in results]
        assert ids[0] != ids[1]
        assert len(ids[0]) == 8  # UUID hex[:8]


class TestE2EMultipleRegistrations:
    """E2E tests for multiple task registrations."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_registered_and_executed(self):
        """Multiple tasks registered and each executed correctly."""
        executor = TaskExecutor()

        async def task_a(query: str) -> str:
            return f"A: {query}"

        async def task_b(query: str) -> str:
            return f"B: {query}"

        async def task_c(query: str) -> str:
            return f"C: {query}"

        executor.register_task(_make_manifest(id="task.a"), task_a)
        executor.register_task(_make_manifest(id="task.b"), task_b)
        executor.register_task(_make_manifest(id="task.c"), task_c)

        # Execute all three
        results = await executor.execute_parallel_async([
            _make_call("task.a", {"query": "x"}),
            _make_call("task.b", {"query": "y"}),
            _make_call("task.c", {"query": "z"}),
        ])

        assert len(results) == 3
        parsed = [json.loads(r.content) for r in results]

        assert parsed[0]["task_type"] == "task.a"
        assert parsed[1]["task_type"] == "task.b"
        assert parsed[2]["task_type"] == "task.c"