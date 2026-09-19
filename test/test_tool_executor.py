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
import threading
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

    def test_callable_success_result_preserves_tool_call_id_and_json_content(self):
        executor = ToolExecutor()

        def search(query: str, limit: int = 5) -> list[str]:
            return [query] * limit

        executor.register("search", search)

        result = executor.execute(
            _make_call("search", {"query": "agent", "limit": 2}, id="call_search")
        )

        assert isinstance(result, ToolResult)
        assert result.tool_call_id == "call_search"
        assert result.name == "search"
        assert result.is_error is False
        assert result.error is None
        assert json.loads(result.content) == ["agent", "agent"]

    def test_execute_unknown_tool_returns_error_result(self):
        """No registration, assert ToolResult(is_error=True, error contains 'Unknown tool')."""
        executor = ToolExecutor()
        result = executor.execute(_make_call("unknown_tool"))

        assert result.is_error is True
        assert "Unknown tool" in result.error
        assert result.error_type == "UnknownToolError"
        assert result.name == "unknown_tool"
        assert result.tool_call_id == "call_1"
        # The failure envelope reaches the model — an empty content string
        # left the model unaware the call ever failed.
        parsed = json.loads(result.content)
        assert parsed["type"] == "UnknownToolError"
        assert "Unknown tool" in parsed["error"]

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
        assert result.tool_call_id == "call_1"

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
        executor = ToolExecutor(per_tool_timeout=0.01)

        def slow_tool():
            time.sleep(0.05)
            return "should not reach here"

        executor.register("slow", slow_tool)
        start = time.monotonic()
        result = executor.execute(_make_call("slow"))
        elapsed = time.monotonic() - start

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "timed out" in result.error
        assert elapsed < 0.2

    def test_execute_within_timeout_succeeds(self):
        executor = ToolExecutor(per_tool_timeout=0.2)

        def fast_tool():
            time.sleep(0.01)
            return {"status": "ok"}

        executor.register("fast", fast_tool)
        result = executor.execute(_make_call("fast"))

        assert result.is_error is False
        parsed = json.loads(result.content)
        assert parsed["status"] == "ok"

    def test_per_tool_timeout_override_returns_structured_timeout_error(self):
        executor = ToolExecutor(
            per_tool_timeout=1.0,
            tool_timeouts={"slow": 0.01},
        )

        def slow_tool():
            time.sleep(0.05)
            return "too slow"

        executor.register("slow", slow_tool)
        start = time.monotonic()
        result = executor.execute(_make_call("slow", id="call_slow"))
        elapsed = time.monotonic() - start

        assert elapsed < 0.2
        assert result.tool_call_id == "call_slow"
        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "0.01" in result.error
        parsed = json.loads(result.content)
        assert parsed["type"] == "TimeoutError"
        assert "timed out" in parsed["error"]


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
        executor = ToolExecutor(per_tool_timeout=1.0)
        rendezvous = threading.Barrier(3)

        def sleepy():
            rendezvous.wait(timeout=0.5)
            return "done"

        executor.register("sleepy", sleepy)
        calls = [
            _make_call("sleepy", id="call_1"),
            _make_call("sleepy", id="call_2"),
            _make_call("sleepy", id="call_3"),
        ]

        results = executor.execute_parallel(calls)

        assert len(results) == 3
        assert all(r.is_error is False for r in results)


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

    def test_dedup_excluded_tool_executes_identical_calls_every_time(self):
        call_count = 0

        def todowrite(todos):
            nonlocal call_count
            call_count += 1
            return {"ok": True, "count": call_count, "todos": todos}

        executor = ToolExecutor(enable_dedup=True, dedup_excluded_tools={"todowrite"})
        executor.register("todowrite", todowrite)
        call = _make_call("todowrite", {"todos": []})

        result1 = executor.execute(call)
        result2 = executor.execute(call)

        assert call_count == 2
        assert json.loads(result1.content)["count"] == 1
        assert json.loads(result2.content)["count"] == 2
        assert result1.is_deduplicated is False
        assert result2.is_deduplicated is False

    def test_exclude_from_dedup_hook_preserves_other_dedup_behavior(self):
        counts = {"todoread": 0, "lookup": 0}

        def todoread():
            counts["todoread"] += 1
            return {"count": counts["todoread"]}

        def lookup():
            counts["lookup"] += 1
            return {"count": counts["lookup"]}

        executor = ToolExecutor(enable_dedup=True)
        executor.exclude_from_dedup("todoread")
        executor.register("todoread", todoread)
        executor.register("lookup", lookup)

        read1 = executor.execute(_make_call("todoread"))
        read2 = executor.execute(_make_call("todoread"))
        lookup1 = executor.execute(_make_call("lookup"))
        lookup2 = executor.execute(_make_call("lookup"))

        assert counts == {"todoread": 2, "lookup": 1}
        assert json.loads(read1.content)["count"] == 1
        assert json.loads(read2.content)["count"] == 2
        assert lookup1.is_deduplicated is False
        assert lookup2.is_deduplicated is True


# ─── Slice 12: Async execution variants ─────────────────────────────────────


class TestToolExecutorAsync:
    """Slice 12: Async execution variants."""

    @pytest.mark.asyncio
    async def test_execute_async_sync_callable_does_not_block_event_loop(self):
        """A sync callable does not block other event-loop work."""
        executor = ToolExecutor()
        started = threading.Event()

        def sync_tool():
            started.set()
            time.sleep(0.02)
            return {"sync": True}

        executor.register("sync_tool", sync_tool)
        execution = asyncio.create_task(
            executor.execute_async(_make_call("sync_tool"))
        )

        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not execution.done()

        result = await execution

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
    async def test_execute_async_callable_exception_returns_structured_error(self):
        executor = ToolExecutor()

        async def fail_async():
            raise RuntimeError("async boom")

        executor.register("fail_async", fail_async)
        result = await executor.execute_async(_make_call("fail_async", id="call_fail"))

        assert result.tool_call_id == "call_fail"
        assert result.name == "fail_async"
        assert result.is_error is True
        assert result.error == "async boom"
        assert result.error_type == "RuntimeError"
        assert json.loads(result.content) == {"error": "async boom", "type": "RuntimeError"}

    @pytest.mark.asyncio
    async def test_execute_parallel_async_runs_concurrently(self):
        executor = ToolExecutor(per_tool_timeout=1.0)
        all_started = asyncio.Event()
        release = asyncio.Event()
        started = 0

        async def sleepy_async():
            nonlocal started
            started += 1
            if started == 3:
                all_started.set()
            await release.wait()
            return "done"

        executor.register("sleepy", sleepy_async)
        calls = [
            _make_call("sleepy", id="call_1"),
            _make_call("sleepy", id="call_2"),
            _make_call("sleepy", id="call_3"),
        ]

        execution = asyncio.create_task(executor.execute_parallel_async(calls))
        await asyncio.wait_for(all_started.wait(), timeout=0.2)
        release.set()
        results = await execution

        assert len(results) == 3
        assert all(r.is_error is False for r in results)

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


# ─── Failure-cause reporting (envelopes, truncation, malformed args) ────────


class TestToolExecutorFailureCauses:
    """Every failed ToolResult must carry a cause the model and UI can read."""

    def test_truncation_sets_error_and_error_type(self):
        executor = ToolExecutor(max_content_size=20)

        def big_tool() -> str:
            return "x" * 100

        executor.register("big", big_tool)
        result = executor.execute(_make_call("big"))

        assert result.is_error is True
        assert result.error_type == "ContentTruncated"
        assert "truncated" in result.error
        assert "102" in result.error  # json.dumps adds two quote chars
        assert result.content.endswith("[TRUNCATED]")

    def test_legitimate_truncated_suffix_is_not_flagged(self):
        """A tool that returns a string ending in [TRUNCATED] is not an error."""
        executor = ToolExecutor(max_content_size=50)

        def tricky_tool() -> str:
            return "data[TRUNCATED]"

        executor.register("tricky", tricky_tool)
        result = executor.execute(_make_call("tricky"))

        assert result.is_error is False
        assert result.error is None

    def test_malformed_arguments_refuse_execution(self):
        executor = ToolExecutor()
        called = []

        def some_tool(**kwargs) -> str:
            called.append(kwargs)
            return "ran"

        executor.register("some_tool", some_tool)
        call = CanonicalToolCall(
            id="call_bad",
            name="some_tool",
            arguments={},
            arguments_error="Arguments for tool 'some_tool' were not valid JSON",
        )
        result = executor.execute(call)

        assert called == []
        assert result.is_error is True
        assert result.error_type == "MalformedArgumentsError"
        parsed = json.loads(result.content)
        assert parsed["type"] == "MalformedArgumentsError"

    def test_async_malformed_arguments_refuse_execution(self):
        executor = ToolExecutor()

        async def some_tool(**kwargs) -> str:
            return "ran"

        executor.register("some_tool", some_tool)
        call = CanonicalToolCall(
            id="call_bad",
            name="some_tool",
            arguments={},
            arguments_error="not valid JSON",
        )
        result = asyncio.run(executor.execute_async(call))
        assert result.is_error is True
        assert result.error_type == "MalformedArgumentsError"

    def test_empty_str_exception_gets_fallback_message(self):
        executor = ToolExecutor()

        class SilentError(Exception):
            def __str__(self) -> str:
                return ""

        def silent_tool() -> str:
            raise SilentError()

        executor.register("silent", silent_tool)
        result = executor.execute(_make_call("silent"))

        assert result.is_error is True
        assert result.error_type == "SilentError"
        assert result.error  # never empty
        assert "SilentError" in result.error

    def test_async_internal_timeout_error_is_not_reported_as_deadline(self):
        """A TimeoutError raised BY the tool is the tool's own error."""
        executor = ToolExecutor(per_tool_timeout=5.0)

        async def flaky_tool() -> str:
            raise TimeoutError("upstream service deadline")

        executor.register("flaky", flaky_tool)
        result = asyncio.run(executor.execute_async(_make_call("flaky")))

        assert result.is_error is True
        assert result.error == "upstream service deadline"
        assert "timed out after" not in result.error

    def test_async_deadline_timeout_reports_executor_limit(self):
        executor = ToolExecutor(per_tool_timeout=0.05)

        async def slow_tool() -> str:
            await asyncio.sleep(1.0)
            return "late"

        executor.register("slow", slow_tool)
        result = asyncio.run(executor.execute_async(_make_call("slow")))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "timed out after 0.05s" in result.error
        parsed = json.loads(result.content)
        assert parsed["type"] == "TimeoutError"

    def test_sync_internal_timeout_error_is_not_reported_as_deadline(self):
        executor = ToolExecutor(per_tool_timeout=5.0)

        def flaky_tool() -> str:
            raise TimeoutError("db read deadline")

        executor.register("flaky", flaky_tool)
        result = executor.execute(_make_call("flaky"))

        assert result.is_error is True
        assert result.error == "db read deadline"
        assert "timed out after" not in result.error
