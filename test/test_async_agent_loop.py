"""Tests for Phase 3: AsyncAgentLoop.

TDD: react-like-agent-loop-refactor — Phase 3 (Slices 10-13)

Covers:
- Slice 10: AsyncAgentLoop constructor, concurrency guard (asyncio.Lock)
- Slice 11: AsyncAgentLoop.run() async ReAct loop
- Slice 12: AsyncAgentLoop.stream() AsyncIterator-only, chunk ordering
- Slice 13: AsyncAgentLoop.state read-only copy
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message, UsageModel
from magic_llm.model.ModelChatStream import (
    ChatCompletionModel,
    ChoiceModel,
    DeltaModel,
)
from magic_llm.agent.types import (
    AgentBudget,
    AgentBudgetExceeded,
    AgentState,
    CanonicalToolCall,
)
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.tool_adapters import ToolAdapter, ToolAdapterFactory
from magic_llm.agent._loop_shared import (
    _build_initial_chat,
    _check_budget,
    _finalize_response,
    _invoke_hook_safely,
    _register_tools_with_executor,
)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _make_response(content=None, tool_calls=None, finish_reason="stop",
                   prompt_tokens=10, completion_tokens=5):
    """Build a valid ModelChatResponse."""
    message = Message(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(index=0, message=message, finish_reason=finish_reason)
    return ModelChatResponse(
        id="test-1",
        object="chat.completion",
        created=1700000000.0,
        model="test-model",
        choices=[choice],
        usage=UsageModel(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_tool_call(id="call_1", name="get_weather",
                    arguments='{"city":"London"}'):
    """Build a valid ToolCall."""
    from magic_llm.model.ModelChatResponse import ToolCall, FunctionCall
    return ToolCall(id=id, function=FunctionCall(name=name, arguments=arguments))


def _make_mock_adapter(is_finished=True, tool_calls=None):
    """Create a mock ToolAdapter."""
    adapter = MagicMock(spec=ToolAdapter)
    adapter.serialize_tool_defs.return_value = None
    adapter.deserialize_tool_calls.return_value = tool_calls or []
    adapter.is_finished.return_value = is_finished
    adapter.extract_final_text.return_value = ""
    adapter.validate_pair_integrity.return_value = True
    adapter.serialize_tool_results.return_value = None
    return adapter


# ─── Slice 10: AsyncAgentLoop constructor, concurrency guard

class TestAsyncAgentLoopConstructor:
    """AsyncAgentLoop.__init__ matches sync constructor."""

    def test_async_loop_constructor_matches_sync(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
            adapter=adapter,
            budget=AgentBudget(max_iterations=5),
            deduplicate=True,
        )
        assert loop._budget.max_iterations == 5
        assert loop._deduplicate is True
        assert loop._adapter is adapter

    def test_async_loop_run_is_coroutine(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)
        assert asyncio.iscoroutinefunction(loop.run)

    def test_async_loop_concurrent_run_raises_runtime_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        async def try_concurrent():
            loop._running = True
            with pytest.raises(RuntimeError, match="already running"):
                await loop.run("hello")

        asyncio.run(try_concurrent())

    def test_async_lock_released_after_completion(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(content="done", finish_reason="stop")
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        async def run_and_check():
            await loop.run("hello")
            assert loop._running is False
            assert not loop._lock.locked()

        asyncio.run(run_and_check())

    def test_async_lock_released_after_exception(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=False, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            adapter=adapter,
            budget=AgentBudget(max_iterations=0),
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded):
                await loop.run("hello")
            assert loop._running is False

        asyncio.run(run_and_check())

    def test_async_and_sync_instances_independent(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()

        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        sync_loop = AgentLoop(client, tools=[], adapter=adapter)
        async_loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        # They should have independent locks (different types)
        import threading
        assert isinstance(sync_loop._lock, type(threading.Lock()))
        assert isinstance(async_loop._lock, asyncio.Lock)

        # Setting one running doesn't affect the other
        sync_loop._running = True
        assert async_loop._running is False


# ─── Slice 11: AsyncAgentLoop.run()

class TestAsyncAgentLoopRun:
    """AsyncAgentLoop.run() mirrors sync AgentLoop with async/await."""

    def test_async_run_no_tools_single_iteration(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(content="hello world", finish_reason="stop")
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        async def run_and_check():
            response = await loop.run("hello")
            assert client.llm.async_generate.call_count == 1
            assert response.content == "hello world"

        asyncio.run(run_and_check())

    def test_async_run_single_tool_call_then_done(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="final answer", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={"city": "London"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather], adapter=adapter)

        async def run_and_check():
            response = await loop.run("What's the weather?")
            assert client.llm.async_generate.call_count == 2
            assert response.content == "final answer"

        asyncio.run(run_and_check())

    def test_async_run_parallel_tool_calls(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc1 = _make_tool_call(id="call_1", name="tool_a")
        tc2 = _make_tool_call(id="call_2", name="tool_b")

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc1, tc2], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        tool_calls = [
            CanonicalToolCall(id="call_1", name="tool_a", arguments={}),
            CanonicalToolCall(id="call_2", name="tool_b", arguments={}),
        ]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def tool_a(): return "a"
        def tool_b(): return "b"

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[tool_a, tool_b], adapter=adapter)

        async def run_and_check():
            response = await loop.run("run both")
            assert client.llm.async_generate.call_count == 2
            assert response.content == "done"

        asyncio.run(run_and_check())

    def test_async_run_budget_exceeded(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(
                content=None, tool_calls=[tc], finish_reason="tool_calls"
            )
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.return_value = tool_calls
        adapter.is_finished.return_value = False
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
            adapter=adapter,
            budget=AgentBudget(max_iterations=2),
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded) as exc_info:
                await loop.run("hello")
            assert exc_info.value.budget_type == "max_iterations"

        asyncio.run(run_and_check())

    def test_async_run_uses_async_generate(self):
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(content="done", finish_reason="stop")
        )
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        async def run_and_check():
            await loop.run("hello")
            assert client.llm.async_generate.called
            # Ensure generate (sync) was NOT called
            assert not client.llm.generate.called

        asyncio.run(run_and_check())

    def test_async_run_uses_execute_parallel_async(self):
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        # Track if execute_parallel_async was called
        original_execute_parallel_async = ToolExecutor.execute_parallel_async

        async def tracked_execute_parallel_async(self, tool_calls):
            tracked_execute_parallel_async.called = True
            return await original_execute_parallel_async(self, tool_calls)

        tracked_execute_parallel_async.called = False

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather], adapter=adapter)

        async def run_and_check():
            with patch.object(
                ToolExecutor, "execute_parallel_async",
                new=tracked_execute_parallel_async,
            ):
                await loop.run("weather?")
            assert tracked_execute_parallel_async.called

        asyncio.run(run_and_check())


# ─── Slice 11b: AsyncAgentLoop.run() with tool_functions

class TestAsyncAgentLoopToolFunctions:
    """AsyncAgentLoop.run() with tool_functions dict and mixed sources."""

    def test_async_run_with_tool_functions_dict(self):
        """tool_functions={"custom_name": fn} is called during async loop execution."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="custom_search", arguments='{"q":"test"}')

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="custom_search", arguments={"q": "test"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        call_log = []

        def custom_search(q):
            call_log.append(q)
            return {"results": [q]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            tool_functions={"custom_search": custom_search},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("search for test")
            assert client.llm.async_generate.call_count == 2
            assert call_log == ["test"]
            assert response.content == "done"

        asyncio.run(run_and_check())

    def test_async_run_mixed_tools_and_tool_functions(self):
        """Both tools=[callable_a] and tool_functions={"custom_b": fn_b} are callable in async loop."""
        client = MagicMock()
        client.llm = MagicMock()
        tc_a = _make_tool_call(id="call_a", name="tool_a")
        tc_b = _make_tool_call(id="call_b", name="custom_b")

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc_a, tc_b], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        tool_calls = [
            CanonicalToolCall(id="call_a", name="tool_a", arguments={}),
            CanonicalToolCall(id="call_b", name="custom_b", arguments={"x": 1}),
        ]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        executed = []

        def tool_a():
            executed.append("tool_a")
            return "a"

        def custom_b(x):
            executed.append(("custom_b", x))
            return {"x": x}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[tool_a],
            tool_functions={"custom_b": custom_b},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("run both")
            assert client.llm.async_generate.call_count == 2
            assert "tool_a" in executed
            assert ("custom_b", 1) in executed
            assert response.content == "done"

        asyncio.run(run_and_check())

    def test_async_run_with_dict_tool_spec_and_tool_functions(self):
        """Dict tool spec in tools list resolves callable from tool_functions in async loop."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the database",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="search_database", arguments='{"query":"climate"}')

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="found results", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="search_database", arguments={"query": "climate"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        call_log = []

        def search_database(query):
            call_log.append(query)
            return {"results": [query]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[tool_spec],
            tool_functions={"search_database": search_database},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("search for climate")
            assert client.llm.async_generate.call_count == 2
            assert call_log == ["climate"]
            assert response.content == "found results"

        asyncio.run(run_and_check())

    def test_async_run_tool_functions_override_callable_with_same_name(self):
        """tool_functions entry overrides callable with same __name__ in async loop."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="get_data")

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="get_data", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def get_data():
            return "from_callable"

        def get_data_override():
            return "from_tool_functions"

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[get_data],
            tool_functions={"get_data": get_data_override},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("test")
            assert response.content == "done"
            serialize_call = adapter.serialize_tool_results.call_args
            results = serialize_call[0][0]
            assert len(results) == 1
            assert "from_tool_functions" in results[0].content
            assert "from_callable" not in results[0].content

        asyncio.run(run_and_check())

    def test_async_run_with_async_tool_function(self):
        """Async callable in tool_functions is awaited during async loop execution."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="async_fetch", arguments='{"url":"https://example.com"}')

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="fetched", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="async_fetch", arguments={"url": "https://example.com"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        call_log = []

        async def async_fetch(url):
            call_log.append(url)
            return {"data": "fetched from " + url}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            tool_functions={"async_fetch": async_fetch},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("fetch data")
            assert client.llm.async_generate.call_count == 2
            assert call_log == ["https://example.com"]
            assert response.content == "fetched"

        asyncio.run(run_and_check())

    def test_async_run_schema_name_mismatch_returns_unknown_tool_error(self):
        """Schema function.name mismatch with tool_functions key produces UnknownToolError."""
        tool_spec = {
            "type": "function",
            "function": {
                "name": "schema_tool_name",
                "description": "A tool",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(name="schema_tool_name", arguments='{"q":"test"}')

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="handled error", finish_reason="stop"),
            ]
        )

        tool_calls = [CanonicalToolCall(id="call_1", name="schema_tool_name", arguments={"q": "test"})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        def different_key_callable(q):
            return {"results": [q]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[tool_spec],
            tool_functions={"different_key": different_key_callable},
            adapter=adapter,
        )

        async def run_and_check():
            response = await loop.run("test mismatch")
            assert response.content == "handled error"
            serialize_call = adapter.serialize_tool_results.call_args
            results = serialize_call[0][0]
            assert len(results) == 1
            assert results[0].is_error is True
            assert "Unknown tool" in results[0].error
            assert results[0].name == "schema_tool_name"

        asyncio.run(run_and_check())


# ─── Slice 12: AsyncAgentLoop.stream()

class TestAsyncAgentLoopStream:
    """AsyncAgentLoop.stream() returns AsyncIterator only."""

    def test_async_stream_returns_async_iterator(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        result = loop.stream("hello")
        # Should be an async generator
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

    def test_async_stream_sync_iteration_raises_type_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        result = loop.stream("hello")
        with pytest.raises(TypeError):
            for _ in result:
                pass

    def test_async_stream_yields_chunks_in_order(self):
        client = MagicMock()
        client.llm = MagicMock()

        chunks = [
            ChatCompletionModel(
                id=f"chunk-{i}",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content=f"c{i}"),
                        finish_reason=None,
                    )
                ],
            )
            for i in range(5)
        ]

        async def async_gen(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client.llm.async_stream_generate = async_gen

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        async def collect_and_check():
            result = []
            async for chunk in loop.stream("hello"):
                result.append(chunk)
            assert len(result) == 5
            for i, chunk in enumerate(result):
                assert chunk.id == f"chunk-{i}"

        asyncio.run(collect_and_check())

    def test_async_stream_uses_initial_chat_multimodal_content(self):
        client = MagicMock()
        client.llm = MagicMock()

        captured = {}
        chunk = ChatCompletionModel(
            id="chunk-mm",
            model="test-model",
            choices=[
                ChoiceModel(
                    index=0,
                    delta=DeltaModel(content="done"),
                    finish_reason="stop",
                )
            ],
        )

        async def async_gen(chat, **kwargs):
            captured["chat_messages"] = chat.messages
            yield chunk

        client.llm.async_stream_generate = async_gen

        adapter = _make_mock_adapter(is_finished=True, tool_calls=[])
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        initial_chat = ModelChat(system="sys")
        initial_chat.add_user_message(
            "Describe this image",
            image=["data:image/png;base64,abc"],
        )

        async def collect_and_check():
            result = []
            async for stream_chunk in loop.stream(initial_chat=initial_chat):
                result.append(stream_chunk)

            assert len(result) == 1
            sent_messages = captured["chat_messages"]
            assert sent_messages[0]["role"] == "system"
            assert sent_messages[1]["role"] == "user"
            assert isinstance(sent_messages[1]["content"], list)
            assert sent_messages[1]["content"][1]["type"] == "image_url"
            assert initial_chat.messages[0]["content"] == "sys"

        asyncio.run(collect_and_check())

    def test_async_stream_yields_separator_between_iterations(self):
        client = MagicMock()
        client.llm = MagicMock()

        # First iteration: tool call
        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )
        chunks_iter0 = [
            ChatCompletionModel(
                id="chunk-0",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="thinking"),
                        finish_reason="tool_calls",
                    )
                ],
            )
        ]
        # Add tool_calls to delta
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(index=0, id="call_1",
                           function=StreamFunctionCall(name="get_weather", arguments="{}"))
        ]

        # Second iteration: done
        chunks_iter1 = [
            ChatCompletionModel(
                id="chunk-1",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="done"),
                        finish_reason="stop",
                    )
                ],
            )
        ]

        iter_count = [0]

        async def async_gen(*args, **kwargs):
            iter_count[0] += 1
            if iter_count[0] == 1:
                for chunk in chunks_iter0:
                    yield chunk
            else:
                for chunk in chunks_iter1:
                    yield chunk

        client.llm.async_stream_generate = async_gen

        tool_calls = [CanonicalToolCall(id="call_1", name="get_weather", arguments={})]
        adapter = MagicMock(spec=ToolAdapter)
        adapter.serialize_tool_defs.return_value = None
        adapter.deserialize_tool_calls.side_effect = [tool_calls, []]
        adapter.is_finished.side_effect = [False, True]
        adapter.extract_final_text.return_value = ""
        adapter.validate_pair_integrity.return_value = True
        adapter.serialize_tool_results.return_value = None

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[lambda: None], adapter=adapter)

        async def collect_and_check():
            result = []
            async for chunk in loop.stream("hello"):
                result.append(chunk)
            # Should have: chunk from iter0 + separator + chunk from iter1
            assert len(result) >= 3
            separator_chunks = [c for c in result if "separator" in c.id]
            assert len(separator_chunks) >= 1

        asyncio.run(collect_and_check())

    def test_async_stream_no_sync_facade(self):
        """AsyncAgentLoop should NOT have a .stream_sync() method."""
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        assert not hasattr(loop, "stream_sync")


# ─── Slice 13: AsyncAgentLoop.state read-only copy

class TestAsyncAgentLoopState:
    """AsyncAgentLoop.state returns independent read-only copies."""

    def test_async_state_property_returns_copy(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        state1 = loop.state
        state1.step = 999
        state2 = loop.state
        assert state2.step != 999

    def test_async_state_messages_are_copied(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        state = loop.state
        state.messages.append({"role": "user", "content": "intruder"})
        assert {"role": "user", "content": "intruder"} not in loop.state.messages

    def test_async_state_independent_from_sync(self):
        client = MagicMock()
        client.llm = MagicMock()
        adapter = _make_mock_adapter()

        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        sync_loop = AgentLoop(client, tools=[], adapter=adapter)
        async_loop = AsyncAgentLoop(client, tools=[], adapter=adapter)

        # States should be independent
        sync_state = sync_loop.state
        async_state = async_loop.state

        sync_state.step = 100
        assert async_state.step != 100


class TestBuildInitialChatInitialChat:
    def test_build_initial_chat_clones_initial_chat(self):
        initial_chat = ModelChat(system="sys")
        initial_chat.add_user_message(
            "Describe this image",
            image=["data:image/png;base64,abc"],
        )

        cloned = _build_initial_chat(initial_chat=initial_chat)

        assert cloned is not initial_chat
        assert cloned.messages == initial_chat.messages
        cloned.messages[0]["content"] = "mutated"
        assert initial_chat.messages[0]["content"] == "sys"
