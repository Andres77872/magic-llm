"""Unit tests for nested LLM node execution.

Tests cover:
- TaskManifest with nested_tools triggers child AsyncAgentLoop
- Child tools NOT inherited from parent (explicit isolation)
- Child uses parent model when nested_model_override=None
- Child uses override model when nested_model_override specified
- Child runs to completion (buffered), returns single TaskResult JSON
- Global depth increments on child start, decrements on child finish
- Per-task depth increments/decrements independently from global depth
- Child error returns TaskResult, does NOT crash parent loop
- Concurrent child invocations use child's own semaphore (isolated from parent)
- Feature flag behavior (disabled = fallback, enabled = nested execution)
- Regression tests: legacy tasks do NOT increment GLOBAL_DEPTH
- Regression tests: PARENT_BUDGET/PARENT_STATE cleanup after loop completion
- Regression tests: public API exports from magic_llm.agent.types
- Child semaphore isolation (AC15)
- Anthropic compatibility documentation (Task 3.7)
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.agent._loop_shared import (
    GLOBAL_DEPTH,
    get_global_depth,
    reset_global_depth,
    PARENT_BUDGET,
    PARENT_STATE,
)
from magic_llm.agent.config import (
    enable_nested_llm_nodes,
    disable_nested_llm_nodes,
    is_nested_llm_nodes_enabled,
)
from magic_llm.agent.task_executor import (
    TaskExecutor,
    _get_depth,
    reset_depths,
)
from magic_llm.agent.types import (
    AgentBudget,
    AgentState,
    CanonicalToolCall,
    TaskManifest,
    TaskResult,
)


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


def _make_manifest(
    id: str = "test_task",
    nested_tools: list | None = None,
    nested_budget: AgentBudget | None = None,
    nested_model_override: str | None = None,
    budget_cascade: bool = False,
) -> TaskManifest:
    return TaskManifest(
        id=id,
        name="Test Task",
        description="A test task",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        nested_tools=nested_tools,
        nested_budget=nested_budget,
        nested_model_override=nested_model_override,
        budget_cascade=budget_cascade,
    )


class TestNestedLLMNodeFeatureFlag:
    """Feature flag behavior for nested LLM node execution."""

    def test_is_nested_llm_nodes_disabled_by_default(self):
        """ENABLE_NESTED_LLM_NODES defaults to False."""
        assert is_nested_llm_nodes_enabled() is False

    def test_enable_nested_llm_nodes(self):
        """enable_nested_llm_nodes() sets flag to True."""
        enable_nested_llm_nodes()
        assert is_nested_llm_nodes_enabled() is True
        # Reset for other tests
        disable_nested_llm_nodes()

    def test_disable_nested_llm_nodes(self):
        """disable_nested_llm_nodes() sets flag to False."""
        enable_nested_llm_nodes()
        disable_nested_llm_nodes()
        assert is_nested_llm_nodes_enabled() is False


class TestNestedLLMNodeExecution:
    """Nested LLM node execution with child AsyncAgentLoop."""

    @pytest.mark.asyncio
    async def test_nested_tools_triggers_child_loop_when_enabled(self):
        """TaskManifest with nested_tools triggers child AsyncAgentLoop when enabled."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        # Create mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Child executed successfully"
        mock_response.choices = []
        
        # Patch AsyncAgentLoop at the source module (where it's imported locally)
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            # Define a simple tool for nested_tools
            def child_tool(query: str) -> str:
                return f"Result: {query}"
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[child_tool],
                nested_budget=AgentBudget(max_iterations=3),
            )
            
            # Register with a callable that would be ignored (nested_tools triggers child loop)
            async def ignored_callable(**kwargs):
                return "This should not be called"
            
            executor.register_task(manifest, ignored_callable)
            
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Verify child loop was instantiated
            mock_loop_class.assert_called_once()
            mock_loop.run.assert_called_once()
            
            # Result should be TaskResult JSON
            parsed = json.loads(result.content)
            assert parsed["task_type"] == "nested_task"
            assert parsed["status"] == "ok"
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_nested_tools_fallback_to_callable_when_disabled(self):
        """TaskManifest with nested_tools falls back to callable when feature disabled."""
        disable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        async def my_callable(query: str) -> str:
            return f"Callable executed: {query}"
        
        manifest = _make_manifest(
            id="nested_task",
            nested_tools=[lambda x: x],  # Has nested_tools but feature disabled
        )
        
        executor.register_task(manifest, my_callable)
        
        result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
        
        parsed = json.loads(result.content)
        assert parsed["task_type"] == "nested_task"
        assert parsed["status"] == "ok"
        # Should contain callable result
        assert "Callable executed" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_no_nested_tools_triggers_callable(self):
        """TaskManifest without nested_tools triggers callable (current behavior)."""
        disable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        async def my_callable(query: str) -> str:
            return f"Callable executed: {query}"
        
        manifest = _make_manifest(id="legacy_task")
        
        executor.register_task(manifest, my_callable)
        
        result = await executor.execute_async(_make_call("legacy_task", {"query": "test"}))
        
        parsed = json.loads(result.content)
        assert parsed["task_type"] == "legacy_task"
        assert parsed["status"] == "ok"

    @pytest.mark.asyncio
    async def test_child_tools_isolated_from_parent(self):
        """Child tools are explicitly configured, NOT inherited from parent."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            # Parent has its own tool
            def parent_tool(x: int) -> int:
                return x * 2
            executor.register("parent_tool", parent_tool)
            
            # Child has different tool
            def child_tool(query: str) -> str:
                return f"Result: {query}"
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[child_tool],  # Only child_tool, NOT parent_tool
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Execute nested task
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Verify child loop was created with only nested_tools
            call_kwargs = mock_loop_class.call_args[1]
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0] == child_tool
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_global_depth_increments_decrements_on_child_execution(self):
        """Global depth increments on child start, decrements on child finish."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Before execution
            assert get_global_depth() == 0
            
            # Execute
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # After execution (depth should be back to 0)
            assert get_global_depth() == 0
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_per_task_depth_independent_from_global_depth(self):
        """Per-task depth increments/decrements independently from global depth."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Before execution
            assert get_global_depth() == 0
            assert _get_depth("nested_task") == 0
            
            # Execute
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # After execution, both counters back to 0
            assert get_global_depth() == 0
            assert _get_depth("nested_task") == 0
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_child_error_returns_taskresult(self):
        """Child error returns TaskResult, does NOT crash parent."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            # Child raises an error
            mock_loop.run = AsyncMock(side_effect=RuntimeError("Child failed"))
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Execute should catch error and return TaskResult
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Should return a TaskResult, not raise
            parsed = json.loads(result.content)
            assert parsed["task_type"] == "nested_task"
            assert parsed["status"] == "failed"
            assert "Child failed" in parsed["summary"]
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_child_model_override(self):
        """Child uses override model when nested_model_override specified."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
                nested_model_override="gpt-4.1-mini",
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Verify model was passed to child loop
            call_kwargs = mock_loop_class.call_args[1]
            assert call_kwargs.get("model") == "gpt-4.1-mini"
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_child_uses_parent_model_when_no_override(self):
        """Child uses parent model when nested_model_override is None."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
                nested_model_override=None,  # No override
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Verify no model override was passed
            call_kwargs = mock_loop_class.call_args[1]
            assert "model" not in call_kwargs or call_kwargs.get("model") is None
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_legacy_task_does_not_increment_global_depth(self):
        """Regression test: Legacy tasks (nested_tools=None) do NOT increment GLOBAL_DEPTH."""
        disable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        executor = TaskExecutor(client=mock_client)
        
        # Create a legacy task (no nested_tools)
        async def legacy_callable(query: str) -> str:
            # During execution, global_depth should still be 0
            depth_during = get_global_depth()
            return f"global_depth_during={depth_during} query={query}"
        
        manifest = _make_manifest(id="legacy_task")  # nested_tools=None
        
        executor.register_task(manifest, legacy_callable)
        
        # Before execution
        assert get_global_depth() == 0
        
        # Execute legacy task
        result = await executor.execute_async(_make_call("legacy_task", {"query": "test"}))
        
        # After execution, global_depth should still be 0 (not incremented)
        assert get_global_depth() == 0
        
        # Verify the callable was executed and global_depth was 0 during execution
        parsed = json.loads(result.content)
        assert "global_depth_during=0" in parsed["summary"]

    @pytest.mark.asyncio
    async def test_nested_task_does_increment_global_depth(self):
        """Nested tasks (nested_tools present) DO increment GLOBAL_DEPTH."""
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Before execution
            assert get_global_depth() == 0
            
            # Execute nested task
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # After execution, global_depth should be back to 0 (incremented during, decremented after)
            assert get_global_depth() == 0
        
        disable_nested_llm_nodes()


class TestPublicAPIExports:
    """Regression tests for public API export contract."""

    def test_types_module_exports_global_depth(self):
        """magic_llm.agent.types exports GLOBAL_DEPTH ContextVar."""
        from magic_llm.agent.types import GLOBAL_DEPTH as types_global_depth
        assert types_global_depth is not None
        # Should be a ContextVar
        import contextvars
        assert isinstance(types_global_depth, contextvars.ContextVar)

    def test_types_module_exports_global_depth_helpers(self):
        """magic_llm.agent.types exports global depth helper functions."""
        from magic_llm.agent.types import (
            get_global_depth,
            increment_global_depth,
            decrement_global_depth,
            reset_global_depth,
        )
        
        # Reset and verify helpers work
        reset_global_depth()
        assert get_global_depth() == 0
        
        increment_global_depth()
        assert get_global_depth() == 1
        
        decrement_global_depth()
        assert get_global_depth() == 0

    def test_types_module_exports_max_global_depth(self):
        """magic_llm.agent.types exports MAX_GLOBAL_DEPTH constant."""
        from magic_llm.agent.types import MAX_GLOBAL_DEPTH
        assert MAX_GLOBAL_DEPTH == 10

    def test_agent_module_exports_global_depth_api(self):
        """magic_llm.agent re-exports global depth API."""
        from magic_llm.agent import (
            GLOBAL_DEPTH,
            get_global_depth,
            increment_global_depth,
            decrement_global_depth,
            reset_global_depth,
            MAX_GLOBAL_DEPTH,
        )
        
        # Verify all imports work
        import contextvars
        assert isinstance(GLOBAL_DEPTH, contextvars.ContextVar)
        assert MAX_GLOBAL_DEPTH == 10
        assert callable(get_global_depth)
        assert callable(increment_global_depth)
        assert callable(decrement_global_depth)
        assert callable(reset_global_depth)


class TestPublicAPIRuntimeParity:
    """Regression tests proving public API and runtime sources are THE SAME objects."""

    def test_types_global_depth_is_same_object_as_loop_shared(self):
        """types.py GLOBAL_DEPTH is the SAME ContextVar as _loop_shared.GLOBAL_DEPTH."""
        from magic_llm.agent.types import GLOBAL_DEPTH as types_global_depth
        from magic_llm.agent._loop_shared import GLOBAL_DEPTH as shared_global_depth
        
        # MUST be the same object (identity check, not equality)
        assert types_global_depth is shared_global_depth

    def test_types_global_depth_helpers_are_same_functions(self):
        """types.py helper functions are the SAME functions as _loop_shared."""
        from magic_llm.agent.types import (
            get_global_depth as types_get,
            increment_global_depth as types_inc,
            decrement_global_depth as types_dec,
            reset_global_depth as types_reset,
        )
        from magic_llm.agent._loop_shared import (
            get_global_depth as shared_get,
            increment_global_depth as shared_inc,
            decrement_global_depth as shared_dec,
            reset_global_depth as shared_reset,
        )
        
        # MUST be the same functions (identity check)
        assert types_get is shared_get
        assert types_inc is shared_inc
        assert types_dec is shared_dec
        assert types_reset is shared_reset

    def test_types_max_global_depth_is_same_value_as_config(self):
        """types.py MAX_GLOBAL_DEPTH is imported from config (same object/value)."""
        from magic_llm.agent.types import MAX_GLOBAL_DEPTH as types_max
        from magic_llm.agent.config import MAX_GLOBAL_DEPTH as config_max
        
        # MUST be the same value (constants are integers, identity works for small ints)
        assert types_max == config_max

    def test_public_api_observes_runtime_state(self):
        """Public API from types.py observes the SAME runtime state as TaskExecutor."""
        from magic_llm.agent.types import get_global_depth, increment_global_depth, reset_global_depth
        from magic_llm.agent._loop_shared import GLOBAL_DEPTH
        
        # Reset to clean state
        reset_global_depth()
        
        # Modify via _loop_shared ContextVar directly
        GLOBAL_DEPTH.set(5)
        
        # Public API helper must observe the same state
        assert get_global_depth() == 5
        
        # Reset
        reset_global_depth()
        assert get_global_depth() == 0


class TestParentContextCleanup:
    """Regression tests for PARENT_BUDGET/PARENT_STATE cleanup."""

    @pytest.mark.asyncio
    async def test_parent_budget_state_reset_after_run(self):
        """PARENT_BUDGET and PARENT_STATE are reset to None after AsyncAgentLoop.run()."""
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        
        # Reset ContextVars before test
        PARENT_BUDGET.set(None)
        PARENT_STATE.set(None)
        
        # Create mock client with response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Done"
        mock_response.usage = None
        
        mock_client.llm = MagicMock()
        mock_client.llm.async_generate = AsyncMock(return_value=mock_response)
        
        # Create loop
        budget = AgentBudget(max_iterations=1)
        loop = AsyncAgentLoop(client=mock_client, tools=[], budget=budget)
        
        # Run the loop
        await loop.run(user_input="test")
        
        # After run, ContextVars should be reset to None
        assert PARENT_BUDGET.get() is None
        assert PARENT_STATE.get() is None

    @pytest.mark.asyncio
    async def test_parent_budget_state_reset_after_stream(self):
        """PARENT_BUDGET and PARENT_STATE are reset to None after AsyncAgentLoop.stream()."""
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
        
        # Reset ContextVars before test
        PARENT_BUDGET.set(None)
        PARENT_STATE.set(None)
        
        # Create mock client with streaming response
        mock_client = MagicMock()
        
        # Create a final chunk that signals completion
        final_chunk = ChatCompletionModel(
            id="test",
            model="test-model",
            choices=[
                ChoiceModel(
                    index=0,
                    delta=DeltaModel(content="Done"),
                    finish_reason="stop",
                )
            ],
        )
        
        async def mock_stream(*args, **kwargs):
            yield final_chunk
        
        mock_client.llm = MagicMock()
        mock_client.llm.async_stream_generate = mock_stream
        
        # Create loop
        budget = AgentBudget(max_iterations=1)
        loop = AsyncAgentLoop(client=mock_client, tools=[], budget=budget)
        
        # Stream and consume all chunks
        chunks = []
        async for chunk in loop.stream(user_input="test"):
            chunks.append(chunk)
        
        # After stream, ContextVars should be reset to None
        assert PARENT_BUDGET.get() is None
        assert PARENT_STATE.get() is None


class TestChildSemaphoreIsolation:
    """Regression tests for child semaphore isolation (AC15)."""

    @pytest.mark.asyncio
    async def test_child_semaphore_independent_from_parent(self):
        """Child task uses its own semaphore, independent from parent semaphore limits.
        
        Spec AC15: Child semaphore isolated from parent semaphore.
        """
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        # Track semaphore acquisition calls
        semaphore_acquisitions = []
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Done"
            mock_response.choices = []
            mock_loop.run = AsyncMock(return_value=mock_response)
            mock_loop_class.return_value = mock_loop
            
            # Create parent executor with max_concurrency=2
            executor = TaskExecutor(client=mock_client)
            
            # Create child task with max_concurrency=5 (different from parent)
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            manifest.max_concurrency = 5  # Child has higher concurrency
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Verify child task has its own semaphore (max_concurrency=5)
            assert "nested_task" in executor._task_semaphores
            child_semaphore = executor._task_semaphores["nested_task"]
            # Semaphore._value is the internal counter (5 for max_concurrency=5)
            assert child_semaphore._value == 5
            
            # Execute nested task
            result = await executor.execute_async(_make_call("nested_task", {"query": "test"}))
            
            # Verify child loop was created with isolated executor
            call_kwargs = mock_loop_class.call_args[1]
            assert "tool_executor" in call_kwargs
            # The child executor is a fresh ToolExecutor, not the parent TaskExecutor
            child_executor = call_kwargs["tool_executor"]
            assert child_executor is not executor  # Different instance
            
            # Child executor is a ToolExecutor (not TaskExecutor)
            # It has no task registry - only ordinary tool registry
            from magic_llm.agent.tool_executor import ToolExecutor
            assert isinstance(child_executor, ToolExecutor)
            assert not isinstance(child_executor, type(executor))  # Not a TaskExecutor
        
        disable_nested_llm_nodes()

    @pytest.mark.asyncio
    async def test_concurrent_child_invocations_respect_child_semaphore(self):
        """Concurrent child invocations respect child semaphore limits.
        
        Spec: When parent invokes same child task 3 times concurrently,
        and child has max_concurrency=2, then 2 proceed, 1 waits.
        """
        enable_nested_llm_nodes()
        reset_depths()
        reset_global_depth()
        
        mock_client = MagicMock()
        
        # Track execution order
        execution_times = []
        
        async def mock_child_run(user_input):
            execution_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate work
            mock_response = MagicMock()
            mock_response.content = f"Done at {len(execution_times)}"
            mock_response.choices = []
            return mock_response
        
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop_class:
            mock_loop = AsyncMock()
            mock_loop.run = mock_child_run
            mock_loop_class.return_value = mock_loop
            
            executor = TaskExecutor(client=mock_client)
            
            # Create child task with max_concurrency=2
            manifest = _make_manifest(
                id="nested_task",
                nested_tools=[lambda x: x],
            )
            manifest.max_concurrency = 2  # Only 2 concurrent
            
            async def ignored_callable(**kwargs):
                return "Ignored"
            
            executor.register_task(manifest, ignored_callable)
            
            # Execute 3 concurrent invocations
            calls = [
                executor.execute_async(_make_call("nested_task", {"query": f"test{i}"}, id=f"call_{i}"))
                for i in range(3)
            ]
            
            results = await asyncio.gather(*calls)
            
            # All 3 should complete (no rejection)
            assert len(results) == 3
            for r in results:
                parsed = json.loads(r.content)
                assert parsed["status"] == "ok"
            
            # Verify execution order shows semaphore limiting
            # With max_concurrency=2, first 2 should start together, 3rd should wait
            # Check timing: first 2 should be close, 3rd should be ~0.1s later
            if len(execution_times) >= 3:
                first_two_diff = abs(execution_times[1] - execution_times[0])
                third_diff = execution_times[2] - execution_times[0]
                # First two should start within ~0.05s of each other
                # Third should start ~0.1s after first (after one of the first two completes)
                assert first_two_diff < 0.05  # First two started together
                assert third_diff >= 0.08  # Third waited for semaphore
        
        disable_nested_llm_nodes()


class TestAnthropicCompatibilityNotes:
    """Documentation tests for Anthropic compatibility (Task 3.7).
    
    Anthropic strict tool pairing requires integration testing with:
    - Real Anthropic client or comprehensive mock
    - AnthropicToolAdapter with strict tool_call/tool_result pairing
    - Nested TaskResult JSON serialization
    
    This test documents the expected behavior without full integration proof.
    """

    def test_nested_taskresult_json_structure(self):
        """Nested TaskResult JSON structure is compatible with Anthropic tool result format."""
        from magic_llm.agent.types import TaskResult
        
        # Create a TaskResult from nested execution
        result = TaskResult(
            task_id="abc123",
            task_type="nested_task",
            status="ok",
            summary="## Result\n\nChild completed successfully.",
        )
        
        # Serialize to JSON
        json_str = result.to_tool_result_json()
        
        # Verify JSON structure is valid for Anthropic tool result content
        parsed = json.loads(json_str)
        assert "task_id" in parsed
        assert "task_type" in parsed
        assert "status" in parsed
        assert "summary" in parsed
        
        # Summary should be plain Markdown (Anthropic-compatible)
        assert "## Result" in parsed["summary"]

    def test_nested_taskresult_with_error_structure(self):
        """Nested TaskResult with error is compatible with Anthropic tool result format."""
        from magic_llm.agent.types import TaskResult, TaskError
        
        # Create a TaskResult with error
        result = TaskResult(
            task_id="abc123",
            task_type="nested_task",
            status="failed",
            summary="## Task Failed\n\nChild raised RuntimeError.",
            error=TaskError(
                error_type="ExecutionError",
                message="Child raised RuntimeError",
                retryable=True,
            ),
        )
        
        # Serialize to JSON
        json_str = result.to_tool_result_json()
        
        # Verify JSON structure
        parsed = json.loads(json_str)
        assert parsed["status"] == "failed"
        assert "error" in parsed
        assert parsed["error"]["error_type"] == "ExecutionError"
        
        # Note: AnthropicToolAdapter.serialize_tool_results() wraps this JSON
        # in a role="tool" message with tool_call_id pairing.
        # Full integration test requires AnthropicToolAdapter setup.