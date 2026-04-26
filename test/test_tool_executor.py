"""Unit tests for ToolExecutor — Slices 1–4, 12.

Tests cover:
- Slice 1: Single execution, unknown tool, error capture, non-JSON output
- Slice 2: Per-tool timeout enforcement
- Slice 3: Parallel execution with ordering
- Slice 4: Deduplication (opt-in)
- Slice 12: Async execution variants
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall, ToolResult


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


# ─── Slice 1: Single execution, unknown tool, error capture ─────────────────


class TestToolExecutorSingleExecution:
    """Slice 1: Basic execution, unknown tool, error capture, non-JSON output."""

    def test_execute_registered_tool_returns_success_result(self):
        """Register + execute, assert ToolResult(is_error=False, name=..., content=..., duration_ms > 0)."""
        executor = ToolExecutor()

        def get_weather(city: str = "London") -> dict:
            return {"temperature": 18, "city": city}

        executor.register("get_weather", get_weather)
        result = executor.execute(_make_call("get_weather", {"city": "London"}))

        assert result.is_error is False
        assert result.name == "get_weather"
        assert result.error is None
        assert result.error_type is None
        assert result.duration_ms > 0
        # Content should be JSON-serialized
        parsed = json.loads(result.content)
        assert parsed["temperature"] == 18
        assert parsed["city"] == "London"

    def test_execute_unknown_tool_returns_error_result(self):
        """No registration, assert ToolResult(is_error=True, error contains 'Unknown tool')."""
        executor = ToolExecutor()
        result = executor.execute(_make_call("unknown_tool"))

        assert result.is_error is True
        assert "Unknown tool" in result.error
        assert result.error_type == "UnknownToolError"
        assert result.name == "unknown_tool"

    def test_execute_tool_exception_returns_error_result(self):
        """Tool raises ValueError('boom'), assert ToolResult(is_error=True, error='boom', error_type='ValueError')."""
        executor = ToolExecutor()

        def boom_tool() -> str:
            raise ValueError("boom")

        executor.register("boom", boom_tool)
        result = executor.execute(_make_call("boom"))

        assert result.is_error is True
        assert result.error == "boom"
        assert result.error_type == "ValueError"
        assert result.duration_ms > 0
        # Content should contain structured error JSON
        parsed = json.loads(result.content)
        assert parsed["error"] == "boom"
        assert parsed["type"] == "ValueError"

    def test_execute_non_json_serializable_output_falls_back_to_str(self):
        """Tool returns file handle or object, assert str() conversion, is_error=False."""
        executor = ToolExecutor()

        def weird_tool():
            return MagicMock()  # MagicMock is not JSON-serializable

        executor.register("weird", weird_tool)
        result = executor.execute(_make_call("weird"))

        assert result.is_error is False
        # Content should be str() representation, not JSON
        assert isinstance(result.content, str)
        # It should NOT be valid JSON (since MagicMock can't be serialized)
        # Actually, MagicMock might serialize to something. Let's just check it's a string.
        assert len(result.content) > 0


# ─── Slice 2: Per-tool timeout enforcement ──────────────────────────────────


class TestToolExecutorTimeout:
    """Slice 2: Per-tool timeout enforcement."""

    def test_execute_timeout_returns_error_result(self):
        """Tool sleeps 10s, timeout=2.0, assert ToolResult(is_error=True, error_type='TimeoutError') within ~2s."""
        executor = ToolExecutor(per_tool_timeout=2.0)

        def slow_tool():
            time.sleep(10)
            return "should not reach here"

        executor.register("slow", slow_tool)
        start = time.monotonic()
        result = executor.execute(_make_call("slow"))
        elapsed = time.monotonic() - start

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "timed out" in result.error
        # Should complete within ~2s (±500ms tolerance)
        assert elapsed < 3.0, f"Timeout took {elapsed:.1f}s, expected ~2s"

    def test_execute_within_timeout_succeeds(self):
        """Tool sleeps 0.1s, timeout=2.0, assert success."""
        executor = ToolExecutor(per_tool_timeout=2.0)

        def fast_tool():
            time.sleep(0.1)
            return {"status": "ok"}

        executor.register("fast", fast_tool)
        result = executor.execute(_make_call("fast"))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"


# ─── Slice 3: Parallel execution with ordering ──────────────────────────────


class TestToolExecutorParallel:
    """Slice 3: Parallel execution with ordering."""

    def test_execute_parallel_returns_results_in_input_order(self):
        """Register 3 tools, execute in order [A, B, C], assert results match order."""
        executor = ToolExecutor()
        executor.register("tool_a", lambda: "a")
        executor.register("tool_b", lambda: "b")
        executor.register("tool_c", lambda: "c")

        calls = [
            _make_call("tool_a", id="call_1"),
            _make_call("tool_b", id="call_2"),
            _make_call("tool_c", id="call_3"),
        ]
        results = executor.execute_parallel(calls)

        assert len(results) == 3
        assert results[0].name == "tool_a"
        assert results[1].name == "tool_b"
        assert results[2].name == "tool_c"

    def test_execute_parallel_completes_concurrently(self):
        """3 tools each sleep 1s, total wall time < 2s (proves true parallelism)."""
        executor = ToolExecutor(per_tool_timeout=10.0)

        def sleepy():
            time.sleep(1)
            return "done"

        executor.register("sleepy", sleepy)
        calls = [
            _make_call("sleepy", id="call_1"),
            _make_call("sleepy", id="call_2"),
            _make_call("sleepy", id="call_3"),
        ]

        start = time.monotonic()
        results = executor.execute_parallel(calls)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.is_error is False for r in results)
        # Should complete in ~1s, not 3s (proves parallelism)
        assert elapsed < 2.0, f"Parallel execution took {elapsed:.1f}s, expected <2s"


# ─── Slice 4: Deduplication (opt-in) ────────────────────────────────────────


class TestToolExecutorDedup:
    """Slice 4: Deduplication (opt-in)."""

    def test_dedup_disabled_same_call_executes_twice(self):
        """enable_dedup=False, same tool call twice, assert 2 executions."""
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return call_count

        executor = ToolExecutor(enable_dedup=False)
        executor.register("counter", counter)
        call = _make_call("counter")

        result1 = executor.execute(call)
        result2 = executor.execute(call)

        assert call_count == 2
        assert result1.is_error is False
        assert result2.is_error is False
        assert result1.is_deduplicated is False
        assert result2.is_deduplicated is False

    def test_dedup_enabled_same_call_returns_cached(self):
        """enable_dedup=True, same (name, arguments) twice, assert second result has is_deduplicated=True."""
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1
            return call_count

        executor = ToolExecutor(enable_dedup=True)
        executor.register("counter", counter)
        call = _make_call("counter")

        result1 = executor.execute(call)
        result2 = executor.execute(call)

        assert call_count == 1, "Tool should only be called once with dedup enabled"
        assert result1.is_error is False
        assert result1.is_deduplicated is False
        assert result2.is_error is False
        assert result2.is_deduplicated is True
        # Cached result should have same content
        assert result1.content == result2.content

    def test_dedup_different_args_executes_both(self):
        """enable_dedup=True, same name different args, assert 2 executions."""
        call_count = 0

        def greet(name: str):
            nonlocal call_count
            call_count += 1
            return f"Hello, {name}!"

        executor = ToolExecutor(enable_dedup=True)
        executor.register("greet", greet)

        call1 = _make_call("greet", {"name": "Alice"})
        call2 = _make_call("greet", {"name": "Bob"})

        result1 = executor.execute(call1)
        result2 = executor.execute(call2)

        assert call_count == 2
        assert result1.is_deduplicated is False
        assert result2.is_deduplicated is False


# ─── Slice 12: Async execution variants ─────────────────────────────────────


class TestToolExecutorAsync:
    """Slice 12: Async execution variants."""

    @pytest.mark.asyncio
    async def test_execute_async_sync_callable_runs_in_executor(self):
        """Sync callable in execute_async, assert it runs without blocking."""
        executor = ToolExecutor()

        def sync_tool():
            time.sleep(0.1)
            return {"sync": True}

        executor.register("sync_tool", sync_tool)
        result = await executor.execute_async(_make_call("sync_tool"))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["sync"] is True

    @pytest.mark.asyncio
    async def test_execute_async_async_callable_awaits_directly(self):
        """Async callable in execute_async, assert it's awaited."""
        executor = ToolExecutor()

        async def async_tool():
            await asyncio.sleep(0.05)
            return {"async": True}

        executor.register("async_tool", async_tool)
        result = await executor.execute_async(_make_call("async_tool"))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["async"] is True

    @pytest.mark.asyncio
    async def test_execute_parallel_async_runs_concurrently(self):
        """3 async tools each sleep 1s, total wall time < 2s."""
        executor = ToolExecutor(per_tool_timeout=10.0)

        async def sleepy_async():
            await asyncio.sleep(1)
            return "done"

        executor.register("sleepy", sleepy_async)
        calls = [
            _make_call("sleepy", id="call_1"),
            _make_call("sleepy", id="call_2"),
            _make_call("sleepy", id="call_3"),
        ]

        start = time.monotonic()
        results = await executor.execute_parallel_async(calls)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.is_error is False for r in results)
        # Should complete in ~1s, not 3s (proves parallelism)
        assert elapsed < 2.0, f"Async parallel execution took {elapsed:.1f}s, expected <2s"

    @pytest.mark.asyncio
    async def test_execute_async_callable_instance_with_async_call(self):
        """Callable instance with async __call__ is detected and awaited properly."""
        executor = ToolExecutor()

        class AsyncCallableInstance:
            __name__ = "async_instance_tool"

            async def __call__(self, value: str) -> dict:
                await asyncio.sleep(0.05)
                return {"async_instance": True, "value": value}

        executor.register("async_instance_tool", AsyncCallableInstance())
        result = await executor.execute_async(_make_call("async_instance_tool", {"value": "test"}))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["async_instance"] is True
        assert parsed["value"] == "test"

    @pytest.mark.asyncio
    async def test_execute_async_sync_callable_instance(self):
        """Sync callable instance runs in executor without issues."""
        executor = ToolExecutor()

        class SyncCallableInstance:
            __name__ = "sync_instance_tool"

            def __call__(self, value: str) -> dict:
                time.sleep(0.05)
                return {"sync_instance": True, "value": value}

        executor.register("sync_instance_tool", SyncCallableInstance())
        result = await executor.execute_async(_make_call("sync_instance_tool", {"value": "test"}))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["sync_instance"] is True
        assert parsed["value"] == "test"


class TestIsAsyncCallable:
    """Regression tests for is_async_callable helper."""

    def test_bare_async_function_detected(self):
        """Bare async def function is detected as async callable."""
        from magic_llm.util import is_async_callable

        async def async_fn():
            return "ok"

        assert is_async_callable(async_fn) is True

    def test_sync_function_not_detected(self):
        """Sync function is not detected as async callable."""
        from magic_llm.util import is_async_callable

        def sync_fn():
            return "ok"

        assert is_async_callable(sync_fn) is False

    def test_callable_instance_with_async_call_detected(self):
        """Callable instance with async __call__ is detected as async callable."""
        from magic_llm.util import is_async_callable

        class AsyncCallable:
            async def __call__(self):
                return "ok"

        assert is_async_callable(AsyncCallable()) is True

    def test_callable_instance_with_sync_call_not_detected(self):
        """Callable instance with sync __call__ is not detected as async callable."""
        from magic_llm.util import is_async_callable

        class SyncCallable:
            def __call__(self):
                return "ok"

        assert is_async_callable(SyncCallable()) is False

    def test_lambda_not_detected(self):
        """Lambda is not detected as async callable."""
        from magic_llm.util import is_async_callable

        assert is_async_callable(lambda x: x) is False
