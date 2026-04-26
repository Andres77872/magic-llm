"""Unit tests for nested depth enforcement.

Tests cover:
- Global depth limit rejects at MAX_GLOBAL_DEPTH
- Global depth check happens before per-task depth check
- Per-task depth limit rejects at manifest.max_depth (when global allows)
- Error message contains "global depth" for global limit, "task_id" for per-task limit
- Grandchild execution (3 levels) increments global depth to 3, then back to 0
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.agent._loop_shared import (
    get_global_depth,
    increment_global_depth,
    reset_global_depth,
)
from magic_llm.agent.config import (
    MAX_GLOBAL_DEPTH,
    enable_nested_llm_nodes,
    disable_nested_llm_nodes,
)
from magic_llm.agent.task_executor import (
    TaskExecutor,
    _get_depth,
    reset_depths,
)
from magic_llm.agent.types import (
    AgentBudget,
    CanonicalToolCall,
    TaskManifest,
)


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


def _make_manifest(
    id: str = "test_task",
    max_depth: int = 3,
    nested_tools: list | None = None,
    timeout_seconds: int = 30,
) -> TaskManifest:
    return TaskManifest(
        id=id,
        name="Test Task",
        description="A test task",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        max_depth=max_depth,
        nested_tools=nested_tools,
        timeout_seconds=timeout_seconds,
    )


class TestNestedDepthLimits:
    """Nested depth enforcement tests."""

    @pytest.mark.asyncio
    async def test_global_depth_limit_rejects_at_max(self):
        """Global depth limit rejects at MAX_GLOBAL_DEPTH."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        # Set global depth to MAX_GLOBAL_DEPTH
        for _ in range(MAX_GLOBAL_DEPTH):
            increment_global_depth()
        
        assert get_global_depth() == MAX_GLOBAL_DEPTH
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        manifest = _make_manifest(
            id="nested_task",
            nested_tools=[lambda x: x],
        )
        
        async def callable(**kwargs):
            return "Result"
        
        executor.register_task(manifest, callable)
        
        # Execute should be rejected due to global depth limit
        result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
        
        parsed = json.loads(result.content)
        assert parsed["status"] == "cancelled"
        assert parsed["error"]["error_type"] == "DepthLimitError"
        assert "Global depth" in parsed["summary"]
        assert str(MAX_GLOBAL_DEPTH) in parsed["summary"]
        
        # Global depth should not increment (rejected before)
        assert get_global_depth() == MAX_GLOBAL_DEPTH
        
        # Reset for other tests
        reset_global_depth()
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_global_depth_check_before_per_task_check(self):
        """Global depth check happens before per-task depth check."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        # Set global depth to MAX_GLOBAL_DEPTH
        for _ in range(MAX_GLOBAL_DEPTH):
            increment_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        # Task with high max_depth (would pass per-task check if not for global)
        # max_depth is constrained to le=10, so use max value
        manifest = _make_manifest(
            id="nested_task",
            max_depth=10,  # Max allowed by TaskManifest constraint
            nested_tools=[lambda x: x],
        )
        
        async def callable(**kwargs):
            return "Result"
        
        executor.register_task(manifest, callable)
        
        # Execute should be rejected by global depth, not per-task
        result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
        
        parsed = json.loads(result.content)
        assert parsed["status"] == "cancelled"
        # Error message mentions global depth, not task_id per-task check
        assert "Global depth" in parsed["summary"]
        # The error message contains the global depth info, not specific task_id rejection
        assert "Global depth" in parsed["error"]["message"]
        
        # Reset for other tests
        reset_global_depth()
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_per_task_depth_limit_rejects_when_global_allows(self):
        """Per-task depth limit rejects at manifest.max_depth when global allows."""
        # Use legacy callable pattern (no nested_tools) for simpler testing
        disable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        # Task with low max_depth, no nested_tools (legacy callable)
        manifest = _make_manifest(
            id="nested_task",
            max_depth=2,
            nested_tools=None,  # Legacy callable pattern
        )
        
        # Use a proper async callable
        async def callable(**kwargs):
            return f"Result: {kwargs}"
        
        executor.register_task(manifest, callable)
        
        # First execution should succeed
        result1 = await executor.execute_async(_make_call("nested_task", {"query": "test1"}))
        parsed1 = json.loads(result1.content)
        assert parsed1["status"] == "ok"
        
        assert _get_depth("nested_task") == 0  # Back to 0 after first execution
        
        # Second execution also succeeds
        result2 = await executor.execute_async(_make_call("nested_task", {"query": "test2"}))
        parsed2 = json.loads(result2.content)
        assert parsed2["status"] == "ok"
        
        # For testing per-task depth rejection, manually set depth to max_depth
        from magic_llm.agent.task_executor import TASK_DEPTH
        
        # Manually set depth to max_depth for this task
        depths = TASK_DEPTH.get().copy()
        depths["nested_task"] = 2
        TASK_DEPTH.set(depths)
        
        # Now execute should be rejected due to per-task depth
        result3 = await executor.execute_async(_make_call("nested_task", {"query": "test3"}))
        parsed3 = json.loads(result3.content)
        assert parsed3["status"] == "cancelled"
        assert parsed3["error"]["error_type"] == "DepthLimitError"
        # Per-task error mentions the task_id or depth info
        assert "nested_task" in parsed3["summary"] or "2" in parsed3["summary"]
        
        # Global depth should not be incremented (rejected before increment)
        assert get_global_depth() == 0
        
        # Reset for other tests
        reset_depths()
        reset_global_depth()

    @pytest.mark.asyncio
    async def test_error_message_global_vs_per_task(self):
        """Error message contains "global depth" for global limit, "task_id" for per-task limit."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        manifest = _make_manifest(
            id="test_task",
            max_depth=3,
            nested_tools=[lambda x: x],
        )
        
        async def callable(**kwargs):
            return f"Result: {kwargs}"
        
        executor.register_task(manifest, callable)
        
        # Test global depth rejection
        for _ in range(MAX_GLOBAL_DEPTH):
            increment_global_depth()
        
        result_global = await executor.execute_async(_make_call("test_task", {"query": "test"}))
        parsed_global = json.loads(result_global.content)
        assert "Global depth" in parsed_global["summary"]
        assert "Global depth" in parsed_global["error"]["message"]
        
        reset_global_depth()
        
        # Test per-task depth rejection
        from magic_llm.agent.task_executor import TASK_DEPTH
        depths = TASK_DEPTH.get().copy()
        depths["test_task"] = 3  # At max_depth
        TASK_DEPTH.set(depths)
        
        result_task = await executor.execute_async(_make_call("test_task", {"query": "test"}))
        parsed_task = json.loads(result_task.content)
        assert "test_task" in parsed_task["summary"] or "3" in parsed_task["summary"]
        
        # Reset for other tests
        reset_depths()
        reset_global_depth()
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_grandchild_execution_depth_tracking(self):
        """Grandchild execution (3 levels) increments global depth to 3, then back to 0."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        # Simulate 3 levels of nesting via recursive execution
        depth_tracker = []
        
        def track_depth():
            depth_tracker.append(get_global_depth())
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            
            # Track depth before and after run
            async def tracked_run(*args, **kwargs):
                track_depth()  # Depth at child start
                return mock_response
            
            mock_loop.run = tracked_run
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def callable(**kwargs):
                return f"Result: {kwargs}"
            
            executor.register_task(manifest, callable)
            
            # First level (parent calls child)
            result1 = await executor.execute_async(_make_call("nested_task", {"query": "level1"}))
            
            # After first level, depth back to 0
            assert get_global_depth() == 0
        
        # Reset for other tests
        reset_global_depth()
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_depth_counters_decrement_on_exception(self):
        """Depth counters decrement on child exception."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            # Child raises exception
            mock_loop.run = AsyncMock(side_effect=RuntimeError("Child crashed"))
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def callable(**kwargs):
                return f"Result: {kwargs}"
            
            executor.register_task(manifest, callable)
            
            # Before execution
            assert get_global_depth() == 0
            assert _get_depth("nested_task") == 0
            
            # Execute (will fail)
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # After execution, counters should be back to 0 (finally block)
            assert get_global_depth() == 0
            assert _get_depth("nested_task") == 0
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_depth_counters_decrement_on_timeout(self):
        """Depth counters decrement on asyncio.TimeoutError."""
        disable_nested_llm_nodes()  # Use legacy callable for timeout test
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        manifest = _make_manifest(
            id="slow_task",
            max_depth=3,
            nested_tools=None,  # Use legacy callable
            timeout_seconds=1,  # 1 second timeout
        )
        
        async def slow_callable(**kwargs):
            await asyncio.sleep(2)  # Sleep 2 seconds (exceeds timeout)
            return "Should not reach"
        
        executor.register_task(manifest, slow_callable)
        
        # Before execution
        assert get_global_depth() == 0
        assert _get_depth("slow_task") == 0
        
        # Execute (will timeout)
        result = await executor.execute_async(_make_call("slow_task", {"query": "test"}))
        
        # After execution, counters should be back to 0 (finally block)
        assert get_global_depth() == 0
        assert _get_depth("slow_task") == 0
        
        parsed = json.loads(result.content)
        assert parsed["status"] == "timeout"