"""Tests for Phase 3: AsyncAgentLoop.

TDD: react-like-agent-loop-refactor — Phase 3 (Slices 10-13)

Covers:
- Slice 10: AsyncAgentLoop constructor, concurrency guard (asyncio.Lock)
- Slice 11: AsyncAgentLoop.run() async ReAct loop
- Slice 12: AsyncAgentLoop.stream() AsyncIterator-only, chunk ordering
- Slice 13: AsyncAgentLoop.state read-only copy
"""

import asyncio
import inspect
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from magic_llm import MagicLLM
from magic_llm.agent import config as agent_config
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
)
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.tool_adapters import ToolAdapter
from magic_llm.agent._loop_shared import (
    _check_budget,
    _finalize_response,
    _invoke_hook_safely,
    _register_tools_with_executor,
)


@pytest.fixture(autouse=True)
def restore_builtin_todo_tools_flag():
    original = agent_config.ENABLE_BUILTIN_TODO_TOOLS
    yield
    agent_config.ENABLE_BUILTIN_TODO_TOOLS = original


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


def _todo(todo_id=1, content="Do work", status="in_progress", priority="high"):
    return {"id": todo_id, "content": content, "status": status, "priority": priority}


def _make_mock_adapter():
    """Create a compatibility adapter for constructor-selection tests."""
    return MagicMock(spec=ToolAdapter)


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

    def test_async_loop_constructor_with_defaults(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])
        assert loop._budget.max_iterations == 150

    def test_async_loop_run_is_coroutine(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])
        assert inspect.iscoroutinefunction(loop.run)

    def test_async_loop_concurrent_run_raises_runtime_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

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
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        async def run_and_check():
            await loop.run("hello")
            assert loop._running is False
            assert not loop._lock.locked()

        asyncio.run(run_and_check())

    def test_async_lock_released_after_exception(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
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

        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        sync_loop = AgentLoop(client, tools=[])
        async_loop = AsyncAgentLoop(client, tools=[])

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
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        async def run_and_check():
            response = await loop.run("hello")
            assert client.llm.async_generate.call_count == 1
            assert response.content == "hello world"

        asyncio.run(run_and_check())


class TestAsyncAgentLoopBuiltinTodoTools:
    def test_async_builtin_schemas_are_injected_by_default(self):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        client = MagicMock()
        client.llm = MagicMock()
        loop = AsyncAgentLoop(client, tools=[])

        assert [tool["function"]["name"] for tool in loop._tools[:2]] == [
            "todowrite",
            "todoread",
        ]

    async def test_async_builtin_write_read_and_state_persistence(self):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[
                        _make_tool_call(
                            id="write_1",
                            name="todowrite",
                            arguments=json.dumps({"todos": [_todo()]}),
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                _make_response(
                    tool_calls=[_make_tool_call(id="read_1", name="todoread", arguments="{}")],
                    finish_reason="tool_calls",
                ),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        loop = AsyncAgentLoop(client, tools=[], budget=AgentBudget(max_iterations=5))
        response = await loop.run("track")

        assert response.content == "done"
        tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
        assert json.loads(tool_messages[0]["content"]) == {"ok": True, "todos": [_todo()]}
        assert json.loads(tool_messages[1]["content"]) == {"ok": True, "todos": [_todo()]}

    def test_async_separate_runs_start_empty(self):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        async def run_and_check():
            client_a = MagicMock()
            client_a.llm = MagicMock()
            client_a.llm.async_generate = AsyncMock(
                side_effect=[
                    _make_response(
                        tool_calls=[
                            _make_tool_call(
                                id="write_a",
                                name="todowrite",
                                arguments=json.dumps({"todos": [_todo(4, "A")]}),
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _make_response(content="done", finish_reason="stop"),
                ]
            )
            await AsyncAgentLoop(client_a, tools=[]).run("first")

            client_b = MagicMock()
            client_b.llm = MagicMock()
            client_b.llm.async_generate = AsyncMock(
                side_effect=[
                    _make_response(
                        tool_calls=[_make_tool_call(id="read_b", name="todoread", arguments="{}")],
                        finish_reason="tool_calls",
                    ),
                    _make_response(content="done", finish_reason="stop"),
                ]
            )
            loop_b = AsyncAgentLoop(client_b, tools=[])
            await loop_b.run("second")
            tool_messages = [m for m in loop_b.state.messages if m.get("role") == "tool"]
            assert json.loads(tool_messages[0]["content"]) == {"ok": True, "todos": []}

        asyncio.run(run_and_check())

    def test_async_reusing_same_loop_starts_next_run_empty(self):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        async def run_and_check():
            client = MagicMock()
            client.llm = MagicMock()
            loop = AsyncAgentLoop(client, tools=[])

            client.llm.async_generate = AsyncMock(
                side_effect=[
                    _make_response(
                        tool_calls=[
                            _make_tool_call(
                                id="write_1",
                                name="todowrite",
                                arguments=json.dumps({"todos": [_todo(9, "First run")]}),
                            )
                        ],
                        finish_reason="tool_calls",
                    ),
                    _make_response(content="done", finish_reason="stop"),
                ]
            )
            await loop.run("first")

            client.llm.async_generate = AsyncMock(
                side_effect=[
                    _make_response(
                        tool_calls=[_make_tool_call(id="read_2", name="todoread", arguments="{}")],
                        finish_reason="tool_calls",
                    ),
                    _make_response(content="done", finish_reason="stop"),
                ]
            )
            await loop.run("second")

            tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
            assert json.loads(tool_messages[0]["content"]) == {"ok": True, "todos": []}

        asyncio.run(run_and_check())

    def test_async_disable_removes_builtin_schemas_and_callables(self):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        agent_config.disable_builtin_todo_tools()
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[_make_tool_call(id="read_1", name="todoread", arguments="{}")],
                    finish_reason="tool_calls",
                ),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        async def run_and_check():
            loop = AsyncAgentLoop(client, tools=[])
            await loop.run("try")
            assert loop._tools == []
            tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
            assert tool_messages[0]["is_error"] is True
            parsed = json.loads(tool_messages[0]["content"])
            assert parsed["type"] == "UnknownToolError"

        asyncio.run(run_and_check())

    def test_async_exact_todoread_collision_warns_and_user_tool_wins(self, caplog):
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(
                    tool_calls=[_make_tool_call(id="read_1", name="todoread", arguments="{}")],
                    finish_reason="tool_calls",
                ),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        def todoread():
            return {"user": True}

        async def run_and_check():
            with caplog.at_level("WARNING"):
                loop = AsyncAgentLoop(client, tools=[todoread])
                await loop.run("read")
            assert "todoread" in caplog.text
            assert "user-provided tool wins" in caplog.text
            tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
            assert json.loads(tool_messages[0]["content"]) == {"user": True}

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

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather])

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

        def tool_a(): return "a"
        def tool_b(): return "b"

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[tool_a, tool_b])

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

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
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
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

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

        # Track if execute_parallel_async was called
        original_execute_parallel_async = ToolExecutor.execute_parallel_async

        async def tracked_execute_parallel_async(self, tool_calls):
            tracked_execute_parallel_async.called = True
            return await original_execute_parallel_async(self, tool_calls)

        tracked_execute_parallel_async.called = False

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather])

        async def run_and_check():
            with patch.object(
                ToolExecutor, "execute_parallel_async",
                new=tracked_execute_parallel_async,
            ):
                await loop.run("weather?")
            assert tracked_execute_parallel_async.called

        asyncio.run(run_and_check())

    # ─── Content Suppression Invariant Tests ────────────────────────────

    def test_async_run_content_suppressed_when_tool_calls(self):
        """INVARIANT + Phase 7: When run() response has both content AND tool_calls,
        content is suppressed for the tool-call iteration — no add_assistant_message
        for THAT iteration. After Phase 7 fix, the final content-only iteration
        correctly records content to state BEFORE break, resulting in 2 assistant
        messages: [tool-call, final-answer]."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content="thinking...", tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="final", finish_reason="stop"),
            ]
        )

        def get_weather(city):
            return {"temp": 18}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather])

        async def run_and_check():
            response = await loop.run("weather?")

            # Count assistant messages in chat history
            assistant_msgs = [
                m for m in loop.state.messages if m.get("role") == "assistant"
            ]
            # Phase 7: Final content-only iteration records content BEFORE break,
            # so we now have 2 assistant messages: [tool-call, final-answer].
            assert len(assistant_msgs) == 2

            # Msg 1: The tool-call message with content=None (pre-tool suppressed)
            tool_call_msg = assistant_msgs[0]
            assert tool_call_msg["role"] == "assistant"
            assert "tool_calls" in tool_call_msg
            assert tool_call_msg.get("content") is None  # pre-tool content suppressed

            # Msg 2: The final answer (no tool_calls, content preserved)
            final_msg = assistant_msgs[1]
            assert final_msg["role"] == "assistant"
            assert "tool_calls" not in final_msg or not final_msg.get("tool_calls")
            assert final_msg.get("content") == "final"

            # The final output is on the returned response
            assert response.content == "final"

        asyncio.run(run_and_check())

    def test_async_run_content_preserved_no_tool_calls(self):
        """INVARIANT: When run() has no tool_calls, content IS preserved
        as normal — normal behavior unaffected."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(content="hello world", finish_reason="stop")
        )

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        async def run_and_check():
            response = await loop.run("hello")

            # Content IS preserved on the returned response
            assert response.content == "hello world"

            # No tool-call messages in chat (no tool_calls existed)
            assert not any(
                "tool_calls" in m for m in loop.state.messages
            )

            # LLM was called exactly once (single iteration, tool-less)
            client.llm.async_generate.assert_called_once()

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

        call_log = []

        def custom_search(q):
            call_log.append(q)
            return {"results": [q]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            tool_functions={"custom_search": custom_search},
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
        tc_a = _make_tool_call(id="call_a", name="tool_a", arguments="{}")
        tc_b = _make_tool_call(id="call_b", name="custom_b", arguments='{"x": 1}')

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc_a, tc_b], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

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

        call_log = []

        def search_database(query):
            call_log.append(query)
            return {"results": [query]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[tool_spec],
            tool_functions={"search_database": search_database},
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
        tc = _make_tool_call(name="get_data", arguments="{}")

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        def get_data():
            return "from_callable"

        def get_data_override():
            return "from_tool_functions"

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[get_data],
            tool_functions={"get_data": get_data_override},
        )

        async def run_and_check():
            response = await loop.run("test")
            assert response.content == "done"
            tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert "from_tool_functions" in tool_messages[0]["content"]
            assert "from_callable" not in tool_messages[0]["content"]

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

        call_log = []

        async def async_fetch(url):
            call_log.append(url)
            return {"data": "fetched from " + url}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            tool_functions={"async_fetch": async_fetch},
        )

        async def run_and_check():
            response = await loop.run("fetch data")
            assert client.llm.async_generate.call_count == 2
            assert call_log == ["https://example.com"]
            assert response.content == "fetched"

        asyncio.run(run_and_check())

    def test_magic_llm_run_agent_async_executes_async_callable_tool_canonically(self):
        """MagicLLM.run_agent_async() preserves async callable registration canonically."""

        class FakeOpenAILLM:
            engine = "openai"

            def __init__(self):
                self.calls = []

            async def async_generate(self, chat, **kwargs):
                self.calls.append((list(chat.messages), kwargs))
                if len(self.calls) == 1:
                    return _make_response(
                        content=None,
                        tool_calls=[
                            _make_tool_call(
                                id="call_weather",
                                name="get_weather_async",
                                arguments='{"city":"Montevideo"}',
                            )
                        ],
                        finish_reason="tool_calls",
                    )
                return _make_response(content="Async weather is 22C.", finish_reason="stop")

        async def get_weather_async(city: str) -> dict:
            """Get async weather for a city."""
            return {"city": city, "temperature_c": 22}

        client = MagicLLM.__new__(MagicLLM)
        client.llm = FakeOpenAILLM()
        client._task_executor = None

        async def run_and_check():
            response = await client.run_agent_async(
                user_input="What's the async weather?",
                tools=[get_weather_async],
                max_iterations=3,
            )

            assert response.content == "Async weather is 22C."
            assert len(client.llm.calls) == 2
            run_tools = client.llm.calls[0][1]["tools"]
            assert [tool["function"]["name"] for tool in run_tools[:2]] == [
                "todowrite",
                "todoread",
            ]
            assert run_tools[2:] == [get_weather_async]
            assert client.llm.calls[0][1]["tool_choice"] == "auto"

            second_messages = client.llm.calls[1][0]
            tool_messages = [m for m in second_messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["tool_call_id"] == "call_weather"
            assert "Montevideo" in tool_messages[0]["content"]

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

        def different_key_callable(q):
            return {"results": [q]}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[tool_spec],
            tool_functions={"different_key": different_key_callable},
        )

        async def run_and_check():
            response = await loop.run("test mismatch")
            assert response.content == "handled error"
            tool_messages = [m for m in loop.state.messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["is_error"] is True
            assert tool_messages[0]["tool_call_id"] == "call_1"

        asyncio.run(run_and_check())


# ─── Slice 12: AsyncAgentLoop.stream()

class TestAsyncAgentLoopStream:
    """AsyncAgentLoop.stream() returns AsyncIterator only."""

    def test_async_stream_returns_async_iterator(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        result = loop.stream("hello")
        # Should be an async generator
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")

    def test_async_stream_sync_iteration_raises_type_error(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

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

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

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

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

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

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[lambda: None])

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
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        assert not hasattr(loop, "stream_sync")

    # ─── Content Suppression Invariant Tests ────────────────────────────

    def test_async_stream_content_suppressed_when_tool_calls(self):
        """INVARIANT: When stream() iteration has content AND tool_calls,
        content is suppressed — no assistant message added, only
        add_tool_call_message(content=None). Pre-tool speculative text
        is never persisted to chat history."""
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )
        # First iteration: content + tool_calls
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
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(index=0, id="call_1",
                           function=StreamFunctionCall(name="get_weather", arguments="{}"))
        ]

        # Second iteration: content only, no tool_calls
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

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[lambda: None])

        async def collect_and_check():
            result = []
            async for chunk in loop.stream("hello"):
                result.append(chunk)

            # Check separator is present (normal stream behavior)
            separator_chunks = [c for c in result if "separator" in c.id]
            assert len(separator_chunks) >= 1

            # INVARIANT: Inspect chat messages after stream completes
            assistant_msgs = [
                m for m in loop.state.messages if m.get("role") == "assistant"
            ]
            # Phase 7 fix: 2 assistant messages — tool-call (iter-1 content suppressed)
            # + final content-only (iter-2 "done" recorded BEFORE break).
            assert len(assistant_msgs) == 2

            tool_call_msg = assistant_msgs[0]
            assert tool_call_msg["role"] == "assistant"
            assert "tool_calls" in tool_call_msg
            assert tool_call_msg.get("content") is None  # iter-1 content suppressed

            # Phase 7: iter-2's content-only "done" IS recorded before break
            final_msg = assistant_msgs[1]
            assert final_msg["role"] == "assistant"
            assert "tool_calls" not in final_msg  # no tool_calls in final answer
            assert final_msg.get("content") == "done"  # final content preserved

        asyncio.run(collect_and_check())

    def test_async_stream_content_preserved_no_tool_calls(self):
        """INVARIANT: When stream() has no tool_calls, content IS preserved
        as normal — all chunks yielded, no content suppression."""
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
                        finish_reason="stop" if i == 4 else None,
                    )
                ],
            )
            for i in range(5)
        ]

        async def async_gen(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        client.llm.async_stream_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        async def collect_and_check():
            result = []
            async for chunk in loop.stream("hello"):
                result.append(chunk)

            # All chunks yielded — content preserved normally
            assert len(result) == 5
            for i, chunk in enumerate(result):
                assert chunk.id == f"chunk-{i}"

            # No separator chunks (no tool calls)
            separator_chunks = [c for c in result if "separator" in c.id]
            assert len(separator_chunks) == 0

            # No tool-call messages in chat
            assert not any(
                "tool_calls" in m for m in loop.state.messages
            )

        asyncio.run(collect_and_check())

    # ─── Phase 7: Tool Result Injection Test ────────────────────────────

    def test_tool_result_present_in_next_llm_call(self):
        """Prove that after tool execution, the NEXT LLM call receives
        role=tool messages with tool_call_id and URL/result content.

        This tests the full injection chain:
          execute_parallel_async -> append_tool_results ->
          chat.messages -> next async_stream_generate(chat)

        Also asserts final assistant message is present in state after
        final content-only response (Phase 7 invariant).

        Rationale: If tool results are NOT injected into chat, the LLM
        never sees tool outputs (URLs, search results, etc.) and the
        next iteration operates blind. This test proves injection is intact.
        """
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )

        # Iter-1: tool_call, no content
        chunks_iter0 = [
            ChatCompletionModel(
                id="chunk-0",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content=""),
                        finish_reason="tool_calls",
                    )
                ],
            )
        ]
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(
                index=0,
                id="call_1",
                function=StreamFunctionCall(name="get_weather", arguments="{}"),
            )
        ]

        # Iter-2: content-only final answer
        chunks_iter1 = [
            ChatCompletionModel(
                id="chunk-1",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="The weather is sunny"),
                        finish_reason="stop",
                    )
                ],
            )
        ]

        # Capture the chat object passed to iter-2's LLM call
        captured_chat_messages: list[list[dict]] = []

        iter_count = [0]

        async def async_gen(*args, **kwargs):
            iter_count[0] += 1
            if iter_count[0] == 1:
                for chunk in chunks_iter0:
                    yield chunk
            else:
                # Capture a COPY of chat.messages from the second LLM call
                chat_arg = args[0]
                captured_chat_messages.append(list(chat_arg.messages))
                for chunk in chunks_iter1:
                    yield chunk

        client.llm.async_stream_generate = async_gen

        # Real tool that returns URL-like data
        def get_weather() -> dict:
            return {"url": "https://api.weather.com/result/sunny"}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[get_weather])

        async def collect_and_check():
            result = []
            async for chunk in loop.stream("What's the weather?"):
                result.append(chunk)

            # ── Assertion 1: Tool result injected into next LLM call ──
            assert len(captured_chat_messages) == 1, (
                "Should have captured chat from iter-2 LLM call"
            )
            iter2_messages = captured_chat_messages[0]

            # Find role=tool messages in iter-2's chat
            tool_msgs = [
                m for m in iter2_messages if m.get("role") == "tool"
            ]
            assert len(tool_msgs) >= 1, (
                "Next LLM call MUST have role=tool messages "
                "from tool results"
            )

            # Verify tool message has tool_call_id and content (URL/result)
            tool_msg = tool_msgs[0]
            assert tool_msg.get("tool_call_id") == "call_1", (
                f"Expected tool_call_id='call_1', "
                f"got {tool_msg.get('tool_call_id')}"
            )
            assert tool_msg.get("content") is not None, (
                "Tool result content MUST NOT be None"
            )
            # The content should contain the URL from the tool result
            # (json.dumps of the dict return value)
            assert "sunny" in tool_msg.get("content", ""), (
                f"Tool result content should contain the URL/result, "
                f"got: {tool_msg.get('content')}"
            )

            # ── Assertion 2: Final assistant message in state ──
            assistant_msgs = [
                m for m in loop.state.messages if m.get("role") == "assistant"
            ]
            assert len(assistant_msgs) >= 2, (
                "Should have tool-call assistant + final answer assistant "
                f"in state, got {len(assistant_msgs)}"
            )

            final_msg = assistant_msgs[-1]
            assert final_msg["role"] == "assistant", (
                f"Last assistant should be final answer, "
                f"got role={final_msg.get('role')}"
            )
            assert "tool_calls" not in final_msg or not final_msg.get("tool_calls"), (
                "Final assistant message MUST NOT have tool_calls"
            )
            assert final_msg.get("content") == "The weather is sunny", (
                f"Final content should be preserved, "
                f"got: {final_msg.get('content')}"
            )

        asyncio.run(collect_and_check())

    def test_magic_llm_run_agent_stream_async_continues_after_tool_call_canonically(self):
        """MagicLLM.run_agent_stream_async() injects canonical tool results before continuing."""
        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )

        class FakeOpenAILLM:
            engine = "openai"

            def __init__(self):
                self.calls = []

            async def async_stream_generate(self, chat, **kwargs):
                self.calls.append((list(chat.messages), kwargs))
                if len(self.calls) == 1:
                    chunk = ChatCompletionModel(
                        id="chunk-tool",
                        model="test-model",
                        choices=[
                            ChoiceModel(
                                index=0,
                                delta=DeltaModel(content=""),
                                finish_reason="tool_calls",
                            )
                        ],
                    )
                    chunk.choices[0].delta.tool_calls = [
                        StreamToolCall(
                            index=0,
                            id="call_weather",
                            function=StreamFunctionCall(
                                name="get_weather_async",
                                arguments='{"city":"Montevideo"}',
                            ),
                        )
                    ]
                    yield chunk
                    return

                yield ChatCompletionModel(
                    id="chunk-final",
                    model="test-model",
                    choices=[
                        ChoiceModel(
                            index=0,
                            delta=DeltaModel(content="Async stream weather used."),
                            finish_reason="stop",
                        )
                    ],
                )

        async def get_weather_async(city: str) -> dict:
            """Get async weather for a city."""
            return {"city": city, "temperature_c": 22}

        client = MagicLLM.__new__(MagicLLM)
        client.llm = FakeOpenAILLM()
        client._task_executor = None

        async def collect_and_check():
            chunks = []
            async for chunk in client.run_agent_stream_async(
                user_input="Stream async weather.",
                tools=[get_weather_async],
                max_iterations=3,
            ):
                chunks.append(chunk)

            assert [chunk.id for chunk in chunks] == ["chunk-tool", "chunk-final"]
            assert chunks[-1].choices[0].delta.content == "Async stream weather used."
            assert len(client.llm.calls) == 2
            stream_tools = client.llm.calls[0][1]["tools"]
            assert [tool["function"]["name"] for tool in stream_tools[:2]] == [
                "todowrite",
                "todoread",
            ]
            assert stream_tools[2:] == [get_weather_async]
            second_messages = client.llm.calls[1][0]
            tool_messages = [m for m in second_messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert tool_messages[0]["tool_call_id"] == "call_weather"
            assert "Montevideo" in tool_messages[0]["content"]

        asyncio.run(collect_and_check())


# ─── Slice 13: AsyncAgentLoop.state read-only copy

class TestAsyncAgentLoopState:
    """AsyncAgentLoop.state returns independent read-only copies."""

    def test_async_state_property_returns_copy(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        state1 = loop.state
        state1.step = 999
        state2 = loop.state
        assert state2.step != 999

    def test_async_state_messages_are_copied(self):
        client = MagicMock()
        client.llm = MagicMock()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        state = loop.state
        state.messages.append({"role": "user", "content": "intruder"})
        assert {"role": "user", "content": "intruder"} not in loop.state.messages

    def test_async_state_independent_from_sync(self):
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.agent.agent_loop import AgentLoop
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        sync_loop = AgentLoop(client, tools=[])
        async_loop = AsyncAgentLoop(client, tools=[])

        # States should be independent
        sync_state = sync_loop.state
        async_state = async_loop.state

        sync_state.step = 100
        assert async_state.step != 100


# ─── Slice 12b: AsyncAgentLoop.stream() budget enforcement

class TestAsyncAgentLoopStreamBudget:
    """AsyncAgentLoop.stream() enforces budgets and fires on_budget_exceeded hook.

    Regression: the inner finally block in stream() unconditionally fired
    on_loop_complete even when AgentBudgetExceeded propagated (the exception
    was caught, on_budget_exceeded was fired, but the re-raise hit the finally
    which also fired on_loop_complete — violating the contract).
    """

    def test_async_stream_budget_exceeded_max_iterations(self):
        """Budget exceeded in async stream raises AgentBudgetExceeded with correct type."""
        client = MagicMock()
        client.llm = MagicMock()

        # Empty async generator — never enters streaming body,
        # pre-call budget check fires immediately
        async def empty_stream(*args, **kwargs):
            return  # no yield = empty async generator
            yield  # pragma: no cover

        client.llm.async_stream_generate = empty_stream

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            budget=AgentBudget(max_iterations=0),
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded) as exc_info:
                async for _ in loop.stream("hello"):
                    pass
            assert exc_info.value.budget_type == "max_iterations"
            assert exc_info.value.current == 0

        asyncio.run(run_and_check())

    def test_async_stream_budget_exceeded_releases_lock(self):
        """Lock is released after budget-exceeded raises in async stream."""
        client = MagicMock()
        client.llm = MagicMock()

        async def empty_stream(*args, **kwargs):
            return
            yield  # pragma: no cover

        client.llm.async_stream_generate = empty_stream

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            budget=AgentBudget(max_iterations=0),
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded):
                async for _ in loop.stream("hello"):
                    pass
            assert loop._running is False

            # Subsequent run with valid budget should work
            loop._budget = AgentBudget(max_iterations=10)
            chunk = ChatCompletionModel(
                id="chunk-1",
                model="test-model",
                choices=[
                    ChoiceModel(
                        index=0,
                        delta=DeltaModel(content="hi"),
                        finish_reason="stop",
                    )
                ],
            )

            async def single_chunk_stream(*args, **kwargs):
                yield chunk

            client.llm.async_stream_generate = single_chunk_stream
            async for _ in loop.stream("hello"):
                pass
            assert loop._running is False

        asyncio.run(run_and_check())

    def test_async_stream_budget_exceeded_does_not_fire_on_loop_complete(self):
        """on_loop_complete hook must NOT fire when budget is exceeded in async stream."""
        client = MagicMock()
        client.llm = MagicMock()

        async def empty_stream(*args, **kwargs):
            return
            yield  # pragma: no cover

        client.llm.async_stream_generate = empty_stream

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        from magic_llm.agent.hooks import AgentHooks

        on_complete = MagicMock()
        on_budget_exceeded = MagicMock()

        hooks = MagicMock(spec=AgentHooks)
        hooks.on_loop_complete = on_complete
        hooks.on_budget_exceeded = on_budget_exceeded

        loop = AsyncAgentLoop(
            client,
            tools=[],
            budget=AgentBudget(max_iterations=0),
            hooks=hooks,
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded):
                async for _ in loop.stream("hello"):
                    pass

            # on_budget_exceeded MUST have been called
            on_budget_exceeded.assert_called_once()
            # on_loop_complete MUST NOT have been called
            on_complete.assert_not_called()

        asyncio.run(run_and_check())

    def test_async_stream_budget_exceeded_wall_clock(self):
        """Wall-clock budget exceeded in async stream raises AgentBudgetExceeded."""
        client = MagicMock()
        client.llm = MagicMock()

        async def slow_stream(*args, **kwargs):
            await asyncio.sleep(0.2)
            return
            yield  # pragma: no cover

        client.llm.async_stream_generate = slow_stream

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            budget=AgentBudget(wall_clock_timeout=0.05),
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded) as exc_info:
                async for _ in loop.stream("hello"):
                    pass
            assert exc_info.value.budget_type == "wall_clock_timeout"

        asyncio.run(run_and_check())


# ─── Budget Hook Recording Helper ────────────────────────────────────────────


class _RecordingBudgetHooks:
    """Simple hook recorder for budget hook tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def on_budget_exceeded(self, budget_type: str, details: str) -> None:
        self.calls.append(("on_budget_exceeded", budget_type, details))


# ─── Phase 5: Budget Hook Activation Tests ──────────────────────────────────


class TestAsyncAgentLoopBudgetHook:
    """on_budget_exceeded fires before AgentBudgetExceeded propagates."""

    def test_budget_hook_fires_on_max_iterations(self):
        """Max iterations exceeded → hook fires → exception re-raised."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(
                content=None, tool_calls=[tc], finish_reason="tool_calls"
            )
        )

        hooks = _RecordingBudgetHooks()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
            budget=AgentBudget(max_iterations=1),
            hooks=hooks,
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded) as exc_info:
                await loop.run("hello")
            assert exc_info.value.budget_type == "max_iterations"
            # Hook must have fired before exception propagated
            assert len(hooks.calls) == 1
            assert hooks.calls[0][0] == "on_budget_exceeded"
            assert hooks.calls[0][1] == "max_iterations"
            assert "limit=1" in hooks.calls[0][2]
            assert "current=1" in hooks.calls[0][2]

        asyncio.run(run_and_check())

    def test_budget_hook_fires_on_token_exceeded(self):
        """Token budget exceeded → hook fires → exception re-raised."""
        client = MagicMock()
        client.llm = MagicMock()
        # Response with completion_tokens=5, but max_output_tokens=1
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(
                content="done", finish_reason="stop",
                completion_tokens=5,
            )
        )

        hooks = _RecordingBudgetHooks()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            budget=AgentBudget(max_output_tokens=1),
            hooks=hooks,
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded) as exc_info:
                await loop.run("hello")
            assert exc_info.value.budget_type == "max_output_tokens"
            assert len(hooks.calls) == 1
            assert hooks.calls[0][0] == "on_budget_exceeded"
            assert hooks.calls[0][1] == "max_output_tokens"
            assert "current=5" in hooks.calls[0][2]

        asyncio.run(run_and_check())

    def test_budget_hook_not_called_on_normal_exit(self):
        """Normal exit within budget → hook NOT called."""
        client = MagicMock()
        client.llm = MagicMock()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(content="hello world", finish_reason="stop")
        )

        hooks = _RecordingBudgetHooks()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            hooks=hooks,
        )

        async def run_and_check():
            response = await loop.run("hello")
            assert response.content == "hello world"
            assert len(hooks.calls) == 0

        asyncio.run(run_and_check())

    def test_budget_hook_receives_budget_type_and_details(self):
        """on_budget_exceeded gets correct budget_type and details."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call()
        client.llm.async_generate = AsyncMock(
            return_value=_make_response(
                content=None, tool_calls=[tc], finish_reason="tool_calls"
            )
        )

        hooks = _RecordingBudgetHooks()
        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
            budget=AgentBudget(max_iterations=1),
            hooks=hooks,
        )

        async def run_and_check():
            with pytest.raises(AgentBudgetExceeded):
                await loop.run("hello")
            # Verify the hook call details
            assert len(hooks.calls) == 1
            call = hooks.calls[0]
            assert call[0] == "on_budget_exceeded"
            assert call[1] == "max_iterations"
            assert isinstance(call[2], str)
            assert len(call[2]) > 0

        asyncio.run(run_and_check())


# ─── Phase 1A: prompt_fragment Tests ──────────────────────────────────────


class TestAsyncAgentLoopPromptFragment:
    """C6-C8: prompt_fragment injection tests for AsyncAgentLoop."""

    def test_c6_static_prompt_fragment_in_system_prompt(self):
        """C6: Static prompt_fragment is prepended to system prompt."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment="Use search first.",
        )

        async def run_and_check():
            await loop.run("hello", system_prompt="You are helpful.")
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert "Use search first." in system_msgs[0]["content"]
            assert system_msgs[0]["content"].startswith("Use search first.")

        asyncio.run(run_and_check())

    def test_c7_callable_prompt_fragment_resolved(self):
        """C7: Callable prompt_fragment is resolved at generation time."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment=lambda: "Dynamic fragment",
        )

        async def run_and_check():
            await loop.run("hello", system_prompt="You are helpful.")
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert "Dynamic fragment" in system_msgs[0]["content"]

        asyncio.run(run_and_check())

    def test_c8_callable_with_kwargs_forwarded(self):
        """C8: Callable prompt_fragment receives forwarded kwargs."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        # url is passed via **kwargs and stored in _generate_kwargs
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment=lambda url: f"URL: {url}",
            url="https://img.jpg",
        )

        async def run_and_check():
            await loop.run("hello", system_prompt="You are helpful.")
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert "URL: https://img.jpg" in system_msgs[0]["content"]

        asyncio.run(run_and_check())

    def test_prompt_fragment_none_does_not_modify_system(self):
        """No prompt_fragment → system prompt is unmodified."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(client, tools=[])

        async def run_and_check():
            await loop.run("hello", system_prompt="You are helpful.")
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert system_msgs[0]["content"] == "You are helpful."

        asyncio.run(run_and_check())

    def test_prompt_fragment_no_system_prompt(self):
        """prompt_fragment with no system_prompt → fragment alone is used."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment="Use tools.",
        )

        async def run_and_check():
            await loop.run("hello", system_prompt=None)
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert system_msgs[0]["content"] == "Use tools."

        asyncio.run(run_and_check())

    # ─────────────────────────────────────────────────────────────────
    # prompt_fragment + initial_chat (Phase 1B blocker regression tests)
    # ─────────────────────────────────────────────────────────────────

    def test_prompt_fragment_with_initial_chat(self):
        """C9: Static prompt_fragment reaches system prompt when initial_chat is provided."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment="Use search first.",
        )

        # Simulate api.magic_llm pattern: prebuilt chat + prompt_fragment
        initial_chat = ModelChat(system="You are a helpful assistant.")
        initial_chat.add_user_message("What is weather in London?")

        async def run_and_check():
            await loop.run(
                user_input=None,
                initial_chat=initial_chat,
            )
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            content = system_msgs[0]["content"]
            assert "Use search first." in content
            assert "You are a helpful assistant." in content
            assert content.startswith("Use search first.")

        asyncio.run(run_and_check())

    def test_prompt_fragment_callable_with_initial_chat(self):
        """C9: Callable prompt_fragment resolved and merged when initial_chat is provided."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment=lambda url: f"Edit URL: {url}",
            url="https://example.com/image.jpg",
        )

        initial_chat = ModelChat(system="You edit images.")
        initial_chat.add_user_message("Edit this image")

        async def run_and_check():
            await loop.run(
                user_input=None,
                initial_chat=initial_chat,
            )
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            content = system_msgs[0]["content"]
            assert "Edit URL: https://example.com/image.jpg" in content
            assert "You edit images." in content
            assert content.startswith("Edit URL:")

        asyncio.run(run_and_check())

    def test_prompt_fragment_with_initial_chat_no_system_in_initial(self):
        """prompt_fragment + initial_chat: inserts at pos 0 when no system msg."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment="PF: instructions.",
        )

        initial_chat = ModelChat()  # no system message
        initial_chat.add_user_message("Direct question")

        async def run_and_check():
            await loop.run(
                user_input=None,
                initial_chat=initial_chat,
            )
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            assert system_msgs[0]["content"] == "PF: instructions."

        asyncio.run(run_and_check())

    def test_prompt_fragment_with_initial_chat_stream(self):
        """C9: prompt_fragment reaches system prompt in stream() with initial_chat."""
        client = MagicMock()
        client.llm = MagicMock()

        captured_chat = {}

        async def async_stream_gen(chat, **kwargs):
            captured_chat["messages"] = list(chat.messages)
            # Return a single no-tool-call streaming response
            yield ChatCompletionModel(
                id="test-1",
                model="test-model",
                choices=[ChoiceModel(
                    index=0,
                    delta=DeltaModel(content="Hello"),
                    finish_reason="stop",
                )],
            )

        client.llm.async_stream_generate = async_stream_gen

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[],
            prompt_fragment="Stream PF.",
        )

        initial_chat = ModelChat(system="System msg.")
        initial_chat.add_user_message("Hi")

        async def run_and_check():
            async for _ in loop.stream(
                user_input=None,
                initial_chat=initial_chat,
            ):
                pass
            system_msgs = [
                m for m in captured_chat["messages"]
                if m.get("role") == "system"
            ]
            assert len(system_msgs) == 1
            content = system_msgs[0]["content"]
            assert "Stream PF." in content
            assert "System msg." in content
            assert content.startswith("Stream PF.")

        asyncio.run(run_and_check())


class TestAsyncAgentLoopEngineTypeWarning:
    """C5: engine_type warning tests for AsyncAgentLoop."""

    def test_engine_type_passed_logs_warning(self):
        """engine_type='anthropic' → logger.warning is called."""
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop

        with patch("magic_llm.agent.async_agent_loop.logger.warning") as mock_warning:
            loop = AsyncAgentLoop(
                client,
                tools=[],
                engine_type="anthropic",
            )
            mock_warning.assert_called_once()
            args, _ = mock_warning.call_args
            warning_fmt = args[0] if args else ""
            # logger.warning receives format string + args separately
            # Check format string contains expected semantics
            assert "ignored" in warning_fmt
            assert "adapter" in warning_fmt or "client.llm" in warning_fmt
            # Check the engine_type value is passed as a formatting arg
            assert len(args) > 1 and args[1] == "anthropic"

        assert "engine_type" not in loop._generate_kwargs


# ─── Phase 6: Per-Iteration prompt_fragment Resolution ────────────────────


class TestAsyncAgentLoopPerIterationPromptFragment:
    """C10-C12: Prompt_fragment is resolved per iteration, not once at init.

    These tests verify the key behavioral change: callable prompt_fragment
    is re-resolved before each LLM call, so document state mutations (or
    any dynamic content) are reflected in each iteration's system prompt.
    """

    def test_run_callable_pf_called_per_iteration(self):
        """C10: Callable prompt_fragment is called once per run() iteration."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(id="call_1", name="get_weather")

        # Two iterations: first with tool calls, second with final answer
        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        # Callable PF that tracks invocation count
        call_count = [0]

        def counting_pf() -> str:
            call_count[0] += 1
            return f"Doc state version {call_count[0]}"

        def get_weather():
            return {"temp": 22}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[get_weather],
            prompt_fragment=counting_pf,
        )

        async def run_and_check():
            await loop.run("weather?", system_prompt="You are helpful.")
            # PF should have been called once per iteration (2 iterations)
            assert call_count[0] == 2, (
                f"Expected 2 PF calls (one per iteration), got {call_count[0]}"
            )

        asyncio.run(run_and_check())

    def test_stream_callable_pf_called_per_iteration(self):
        """C11: Callable prompt_fragment is called once per stream() iteration."""
        client = MagicMock()
        client.llm = MagicMock()

        from magic_llm.model.ModelChatStream import (
            ToolCall as StreamToolCall,
            FunctionCall as StreamFunctionCall,
        )

        # Iteration 1: tool call
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
        chunks_iter0[0].choices[0].delta.tool_calls = [
            StreamToolCall(
                index=0, id="call_1",
                function=StreamFunctionCall(name="get_weather", arguments="{}"),
            )
        ]

        # Iteration 2: final answer
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

        # Callable PF that tracks invocation count
        call_count = [0]

        def counting_pf() -> str:
            call_count[0] += 1
            return f"Doc state version {call_count[0]}"

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[lambda: None],
            prompt_fragment=counting_pf,
        )

        async def collect_and_check():
            async for _ in loop.stream("hello"):
                pass
            # PF should have been called once per iteration (2 iterations)
            assert call_count[0] == 2, (
                f"Expected 2 PF calls (one per iteration), got {call_count[0]}"
            )

        asyncio.run(collect_and_check())

    def test_run_callable_pf_sees_different_values_per_iteration(self):
        """C12: Each iteration's PF call can return different content."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(id="call_1", name="get_weather")

        # Capture the system messages seen by each LLM call
        captured_system_messages: list[str] = []

        async def async_gen_capture(chat, **kwargs):
            for msg in chat.messages:
                if msg.get("role") == "system":
                    captured_system_messages.append(msg["content"])
                    break
            if len(captured_system_messages) == 1:
                return _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls")
            return _make_response(content="done", finish_reason="stop")

        client.llm.async_generate = async_gen_capture

        # Stateful PF that simulates document version changes
        doc_state = {"version": 1}

        def doc_pf() -> str:
            v = doc_state["version"]
            doc_state["version"] += 1
            return f"Document version {v}"

        # Simulate tool that updates doc_ref
        def get_weather():
            return {"temp": 22}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[get_weather],
            prompt_fragment=doc_pf,
        )

        async def run_and_check():
            response = await loop.run("weather?", system_prompt="You are helpful.")
            assert response.content == "done"
            assert captured_system_messages == [
                "Document version 1\n\nYou are helpful.",
                "Document version 2\n\nYou are helpful.",
            ]

        asyncio.run(run_and_check())

    def test_static_prompt_fragment_backward_compat(self):
        """Static prompt_fragment still works identically in multi-iteration run()."""
        client = MagicMock()
        client.llm = MagicMock()
        tc = _make_tool_call(id="call_1", name="get_weather")

        client.llm.async_generate = AsyncMock(
            side_effect=[
                _make_response(content=None, tool_calls=[tc], finish_reason="tool_calls"),
                _make_response(content="done", finish_reason="stop"),
            ]
        )

        captured_first: list[str] = []
        captured_second: list[str] = []
        call_idx = [0]

        async def capturing_gen(chat, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            for msg in chat.messages:
                if msg.get("role") == "system":
                    if idx == 0:
                        captured_first.append(msg["content"])
                    else:
                        captured_second.append(msg["content"])
                    break
            return _make_response(
                content=None if idx == 0 else "done",
                tool_calls=[tc] if idx == 0 else None,
                finish_reason="tool_calls" if idx == 0 else "stop",
            )

        client.llm.async_generate = capturing_gen

        def get_weather():
            return {"temp": 22}

        from magic_llm.agent.async_agent_loop import AsyncAgentLoop
        loop = AsyncAgentLoop(
            client,
            tools=[get_weather],
            prompt_fragment="Static context.",
        )

        async def run_and_check():
            await loop.run("test", system_prompt="You are helpful.")
            # Both iterations should have the same static PF
            assert len(captured_first) == 1
            assert len(captured_second) == 1
            assert captured_first[0] == "Static context.\n\nYou are helpful."
            assert captured_second[0] == "Static context.\n\nYou are helpful."

        asyncio.run(run_and_check())
