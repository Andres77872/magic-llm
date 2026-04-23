"""Unit tests for MagicLLM.register_task() API.

Tests cover:
- Lazy TaskExecutor initialization
- Task registration flow
- Integration with run_agent_async()
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from magic_llm import MagicLLM
from magic_llm.agent import TaskManifest, TaskExecutor


def _make_manifest(
    id: str = "test.task",
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


class TestMagicLLMRegisterTaskLazyInit:
    """Tests for lazy TaskExecutor initialization."""

    def test_no_task_executor_before_registration(self):
        """_task_executor is None before any task registration."""
        client = MagicLLM(engine="openai", model="gpt-4")

        assert client._task_executor is None

    def test_lazy_init_on_first_registration(self):
        """_task_executor is created on first register_task call."""
        client = MagicLLM(engine="openai", model="gpt-4")

        # Initially None
        assert client._task_executor is None

        # Register a task triggers lazy init
        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest()
        client.register_task(manifest, my_task)

        # Now executor exists
        assert client._task_executor is not None
        assert isinstance(client._task_executor, TaskExecutor)

    def test_same_executor_reused_for_multiple_tasks(self):
        """Same TaskExecutor instance reused for subsequent registrations."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def task_a(query: str) -> str:
            return f"A: {query}"

        async def task_b(query: str) -> str:
            return f"B: {query}"

        client.register_task(_make_manifest(id="task_a"), task_a)
        executor_ref = client._task_executor

        client.register_task(_make_manifest(id="task_b"), task_b)

        # Same instance reused
        assert client._task_executor is executor_ref


class TestMagicLLMRegisterTaskFlow:
    """Tests for task registration flow."""

    def test_task_registered_in_executor(self):
        """Registered task appears in TaskExecutor registry."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task")
        client.register_task(manifest, my_task)

        # Task appears in executor registry
        assert "my_task" in client._task_executor._task_registry
        stored_manifest = client._task_executor.get_task_manifest("my_task")
        assert stored_manifest.id == "my_task"

    def test_manifest_policy_stored(self):
        """Manifest policy fields are stored correctly."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(
            id="custom_task",
            timeout_seconds=60,
            max_concurrency=3,
            max_depth=2,
        )
        client.register_task(manifest, my_task)

        stored = client._task_executor.get_task_manifest("custom_task")
        assert stored.timeout_seconds == 60
        assert stored.max_concurrency == 3
        assert stored.max_depth == 2

    def test_semaphore_created(self):
        """Semaphore is created for task concurrency control."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task", max_concurrency=5)
        client.register_task(manifest, my_task)

        # Semaphore exists
        assert "my_task" in client._task_executor._task_semaphores
        semaphore = client._task_executor._task_semaphores["my_task"]
        assert semaphore._value == 5

    def test_callable_wrapped_and_registered(self):
        """Callable is wrapped and registered in base registry."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return query

        manifest = _make_manifest(id="my_task")
        client.register_task(manifest, my_task)

        # Callable wrapped and in base registry
        assert "my_task" in client._task_executor._registry


class TestMagicLLMRegisterTaskWithSyncCallable:
    """Tests for registering sync callables (converted to async)."""

    def test_sync_callable_accepted(self):
        """Sync callable is accepted (wrapped by TaskExecutor)."""
        client = MagicLLM(engine="openai", model="gpt-4")

        # Sync callable
        def sync_task(query: str) -> str:
            return f"Sync: {query}"

        manifest = _make_manifest(id="sync_task")
        client.register_task(manifest, sync_task)

        # Task registered
        assert "sync_task" in client._task_executor._task_registry


class TestMagicLLMRegisterTaskEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_multiple_registrations_same_id_replaces(self):
        """Registering same ID twice replaces the previous registration."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def task_v1(query: str) -> str:
            return f"v1: {query}"

        async def task_v2(query: str) -> str:
            return f"v2: {query}"

        manifest = _make_manifest(id="my_task")
        client.register_task(manifest, task_v1)

        # Register again with same ID
        client.register_task(manifest, task_v2)

        # Should have replaced (only one entry)
        registered = client._task_executor.get_registered_tasks()
        assert len([t for t in registered if t == "my_task"]) == 1

    def test_registration_with_different_manifest_same_id(self):
        """Different manifest for same ID updates policy."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return query

        # First registration with timeout=30
        manifest_v1 = _make_manifest(id="my_task", timeout_seconds=30)
        client.register_task(manifest_v1, my_task)

        # Second registration with timeout=60
        manifest_v2 = _make_manifest(id="my_task", timeout_seconds=60)
        client.register_task(manifest_v2, my_task)

        # Policy updated
        stored = client._task_executor.get_task_manifest("my_task")
        assert stored.timeout_seconds == 60


class TestMagicLLMRegisterTaskIntegration:
    """Tests for integration with run_agent_async()."""

    @pytest.mark.asyncio
    async def test_task_executor_passed_to_agent_loop(self):
        """Internal TaskExecutor is passed to AsyncAgentLoop when no override provided."""
        client = MagicLLM(engine="openai", model="gpt-4")

        async def my_task(query: str) -> str:
            return f"Result: {query}"

        manifest = _make_manifest(id="my_task")
        client.register_task(manifest, my_task)

        # Mock the AsyncAgentLoop to capture the executor
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=MagicMock(content="done"))
            mock_loop.return_value = mock_instance

            await client.run_agent_async(user_input="Test")

            # Verify executor was passed
            call_kwargs = mock_loop.call_args[1]
            assert call_kwargs["tool_executor"] is client._task_executor

    @pytest.mark.asyncio
    async def test_explicit_executor_overrides_internal(self):
        """Explicit task_executor overrides internal _task_executor."""
        client = MagicLLM(engine="openai", model="gpt-4")

        # Register task in internal executor
        async def internal_task(query: str) -> str:
            return "internal"

        client.register_task(_make_manifest(id="internal"), internal_task)

        # Create explicit executor with different task
        explicit_executor = TaskExecutor()
        async def explicit_task(query: str) -> str:
            return "explicit"

        explicit_executor.register_task(_make_manifest(id="explicit"), explicit_task)

        # Mock the AsyncAgentLoop
        with patch("magic_llm.agent.async_agent_loop.AsyncAgentLoop") as mock_loop:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=MagicMock(content="done"))
            mock_loop.return_value = mock_instance

            await client.run_agent_async(
                user_input="Test",
                task_executor=explicit_executor,
            )

            # Verify explicit executor was passed
            call_kwargs = mock_loop.call_args[1]
            assert call_kwargs["tool_executor"] is explicit_executor
            assert call_kwargs["tool_executor"] is not client._task_executor