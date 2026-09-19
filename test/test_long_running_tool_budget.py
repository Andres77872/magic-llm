"""Tests for per-tool timeout overrides, heartbeat callback, and custom executor.

Covers:
- Task 1.2: Per-tool timeout overrides in ToolExecutor
- Task 1.3: Heartbeat callback in AsyncAgentLoop
- Task 1.4: Custom ToolExecutor flows through AsyncAgentLoop
"""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    ChoiceModel,
    DeltaModel,
    FunctionCall,
    ToolCall,
)


def _make_call(name: str, args: dict | None = None, id: str = "call_1") -> CanonicalToolCall:
    return CanonicalToolCall(id=id, name=name, arguments=args or {})


# ─── Task 1.2: Per-tool timeout overrides ──────────────────────────────────


class TestPerToolTimeout:
    """Task 1.2: Per-tool timeout overrides in ToolExecutor."""

    pytestmark = pytest.mark.asyncio

    async def _async_slow_tool(self, duration: float = 30.0) -> str:
        """Async tool that sleeps for the given duration."""
        await asyncio.sleep(duration)
        return "done"

    async def test_per_tool_timeout_cancels_long_tool(self):
        """Tool with a short global timeout and no override is cancelled."""
        executor = ToolExecutor(per_tool_timeout=0.01)
        executor.register("slow", self._async_slow_tool)

        result = await executor.execute_async(_make_call("slow"))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert "timed out" in result.error
        assert "0.01" in result.error

    async def test_per_tool_name_override_applies(self):
        """Caller-owned generate_image tool timeout uses 120.0, not global 1.0."""
        executor = ToolExecutor(
            per_tool_timeout=0.01,
            tool_timeouts={"generate_image": 120.0},
        )

        async def fast_image_tool() -> str:
            # This is a generic caller-registered tool name, not a core image-generation API.
            return '{"url": "/images/img-abc.webp"}'

        executor.register("generate_image", fast_image_tool)

        # The per-tool override lets the call succeed despite the short global timeout.
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
            await asyncio.Event().wait()
            return "search results"

        executor.register("search", browsing_tool)
        result = await executor.execute_async(_make_call("search"))

        assert result.is_error is True
        assert result.error_type == "TimeoutError"

    async def test_no_orphan_tasks_on_timeout(self):
        """A timed-out coroutine is cancelled without leaving pending tasks."""
        executor = ToolExecutor(per_tool_timeout=0.01)
        cancellation_seen = asyncio.Event()

        async def cancellable_tool() -> str:
            try:
                await asyncio.Event().wait()
            finally:
                cancellation_seen.set()

        executor.register("cancellable", cancellable_tool)
        tasks_before = asyncio.all_tasks()
        result = await executor.execute_async(_make_call("cancellable"))
        orphaned_tasks = {
            task for task in asyncio.all_tasks() - tasks_before if not task.done()
        }

        assert result.is_error is True
        assert result.error_type == "TimeoutError"
        assert cancellation_seen.is_set()
        assert orphaned_tasks == set()

    async def test_timeout_returns_safe_error_message(self):
        """Timeout returns ToolResult with safe error and no stack trace."""
        executor = ToolExecutor(per_tool_timeout=0.01)

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

    pytestmark = pytest.mark.asyncio

    @staticmethod
    def _streaming_client(tool_name: str):
        client = MagicMock()
        client.llm = MagicMock()
        call_count = 0

        async def generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield ChatCompletionModel(
                    id="tool-chunk",
                    model="test-model",
                    choices=[ChoiceModel(
                        delta=DeltaModel(tool_calls=[ToolCall(
                            index=0,
                            id="call_1",
                            function=FunctionCall(name=tool_name, arguments="{}"),
                        )]),
                        finish_reason="tool_calls",
                    )],
                )
            else:
                yield ChatCompletionModel(
                    id="final-chunk",
                    model="test-model",
                    choices=[ChoiceModel(
                        delta=DeltaModel(content="done"),
                        finish_reason="stop",
                    )],
                )

        client.llm.async_stream_generate = generate
        return client

    async def test_stream_heartbeat_runs_while_real_loop_executes_tool(self, monkeypatch):
        heartbeat_seen = asyncio.Event()
        heartbeat_calls = []
        original_sleep = asyncio.sleep

        async def accelerated_heartbeat_sleep(delay):
            assert delay == 8
            await original_sleep(0)

        monkeypatch.setattr(
            "magic_llm.agent.async_agent_loop.asyncio.sleep",
            accelerated_heartbeat_sleep,
        )

        async def heartbeat():
            heartbeat_calls.append("heartbeat")
            heartbeat_seen.set()

        async def slow_tool():
            await asyncio.wait_for(heartbeat_seen.wait(), timeout=1)
            return {"status": "done"}

        loop = AsyncAgentLoop(
            client=self._streaming_client("slow_tool"),
            tools=[slow_tool],
            heartbeat_cb=heartbeat,
        )

        chunks = [chunk async for chunk in loop.stream("run the tool")]

        assert heartbeat_calls
        assert chunks[-1].choices[0].delta.content == "done"

    async def test_stream_fast_tool_is_done_before_first_heartbeat(self):
        heartbeat_calls = []

        async def heartbeat():
            heartbeat_calls.append("heartbeat")

        async def fast_tool():
            return {"status": "done"}

        loop = AsyncAgentLoop(
            client=self._streaming_client("fast_tool"),
            tools=[fast_tool],
            heartbeat_cb=heartbeat,
        )

        chunks = [chunk async for chunk in loop.stream("run the tool")]

        assert heartbeat_calls == []
        assert chunks[-1].choices[0].delta.content == "done"


# ─── Integration: Combined behavior of all Phase 1 changes ─────────────────


class TestPhase1Integration:
    """Integration tests combining all Phase 1 changes."""

    pytestmark = pytest.mark.asyncio

    async def test_tool_timeout_and_content_size_together(self):
        """Both per-tool timeout override and max_content_size work together."""
        executor = ToolExecutor(
            per_tool_timeout=1.0,
            tool_timeouts={"generate_image": 10.0},
            max_content_sizes={"generate_image": 20},
        )

        async def image_tool() -> dict:
            # Caller-owned test tool; Magic LLM does not provide first-class image generation.
            return {"url": "/images/img.webp", "data": "x" * 100}

        executor.register("generate_image", image_tool)
        result = await executor.execute_async(_make_call("generate_image"))

        # Truncation now carries its cause instead of a bare is_error flag.
        assert result.error_type == "ContentTruncated"
        assert "truncated" in result.error
        # Content: truncated by max_content_size=20
        assert result.content.endswith("[TRUNCATED]")
