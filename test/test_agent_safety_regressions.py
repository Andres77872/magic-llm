"""Behavioral regressions for cancellation, tool ordering and delegated budgets."""
import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from magic_llm.agent import config
from magic_llm.agent._loop_shared import PARENT_BUDGET, PARENT_STATE, _compute_child_budget
from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.builtin_tools import TodoState
from magic_llm.agent.task_executor import TaskExecutor, get_all_depths
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import AgentBudget, AgentBudgetExceeded, AgentState, CanonicalToolCall, TaskManifest
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
from magic_llm.util.async_bridge import run_sync_in_thread


def response(content="done", tokens=1):
    return ModelChatResponse(id="response", object="chat.completion", created=0.0,
        model="fake", choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
        usage=UsageModel(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2))


def client_for(generate=None):
    client = MagicMock()
    client.llm.async_generate = generate or AsyncMock(return_value=response())
    return client


def call(name="work", arguments=None, id="call"):
    return CanonicalToolCall(id=id, name=name, arguments=arguments or {})


def manifest(**kwargs):
    return TaskManifest(id="work", name="Worker", description="Research worker",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"], "additionalProperties": False}, **kwargs)


@pytest.mark.asyncio
async def test_concurrent_run_cannot_reset_live_state_tools_or_parent_context():
    started, finish = asyncio.Event(), asyncio.Event()
    async def generate(*args, **kwargs):
        started.set()
        await finish.wait()
        return response()
    loop = AsyncAgentLoop(client_for(generate))
    running = asyncio.create_task(loop.run("first", system_prompt="original"))
    await started.wait()
    original_state = loop._state
    original_todos = loop._builtin_tool_functions
    parent = PARENT_STATE.get()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await loop.run("second", system_prompt="intruder")
        assert loop._state is original_state
        assert loop._builtin_tool_functions is original_todos
        assert loop._base_system_prompt == "original"
        assert PARENT_STATE.get() is parent
    finally:
        finish.set()
        await running


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_wall_deadline_cancels_stalled_provider(stream):
    cancelled = asyncio.Event()
    async def stall(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    async def chunks(*args, **kwargs):
        await stall()
        yield  # pragma: no cover
    client = client_for(stall)
    client.llm.async_stream_generate = chunks
    hooks = MagicMock()
    loop = AsyncAgentLoop(client, budget=AgentBudget(wall_clock_timeout=.03), hooks=hooks)
    with pytest.raises(AgentBudgetExceeded) as failure:
        if stream:
            async for _ in loop.stream("test"):
                pass
        else:
            await loop.run("test")
    assert failure.value.budget_type == "wall_clock_timeout"
    assert cancelled.is_set()
    assert not loop._running
    hooks.on_budget_exceeded.assert_called_once()
    hooks.on_loop_complete.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_batch_cancels_sibling_tools():
    sibling_started, sibling_cancelled = asyncio.Event(), asyncio.Event()
    async def cancelled():
        await sibling_started.wait()
        raise asyncio.CancelledError
    async def sibling():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()
    executor = ToolExecutor()
    executor.register("cancelled", cancelled)
    executor.register("sibling", sibling)
    with pytest.raises(asyncio.CancelledError):
        await executor.execute_parallel_async([call("cancelled"), call("sibling")])
    assert sibling_cancelled.is_set()


@pytest.mark.asyncio
async def test_tool_parallelism_bounded_and_mutation_is_ordered_barrier():
    executor = ToolExecutor(max_parallel_tools=2, serial_tools={"write"})
    active = 0
    peak = 0
    values = []
    async def read():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.005)
        active -= 1
        return list(values)
    async def write():
        assert active == 0
        values.append("updated")
    executor.register("read", read)
    executor.register("write", write)
    results = await executor.execute_parallel_async(
        [call("read", id=str(i)) for i in range(5)] + [call("write"), call("read")])
    assert peak == 2
    assert all(json.loads(result.content) == [] for result in results[:5])
    assert json.loads(results[-1].content) == ["updated"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dedup_preserves_call_ids_and_refreshes_replaced_tools(asynchronous):
    executor = ToolExecutor(enable_dedup=True)
    calls = []
    def work():
        calls.append(1)
        return "old"
    executor.register("work", work)
    batch = [call(id="first"), call(id="second")]
    results = asyncio.run(executor.execute_parallel_async(batch)) if asynchronous else executor.execute_parallel(batch)
    assert [result.tool_call_id for result in results] == ["first", "second"]
    assert len(calls) == 1
    assert results[1].is_deduplicated
    executor.register("work", lambda: "new")
    assert json.loads(executor.execute(call()).content) == "new"


@pytest.mark.asyncio
async def test_task_refuses_malformed_or_schema_invalid_inputs_and_marks_failures():
    executor = TaskExecutor()
    work = AsyncMock(return_value="should not run")
    executor.register_task(manifest(), work)
    malformed = call(arguments={"query": "ignored"})
    malformed.arguments_error = "invalid JSON"
    assert (await executor.execute_async(malformed)).error_type == "MalformedArgumentsError"
    result = await executor.execute_async(call(arguments={"query": 42}))
    assert result.is_error and result.error_type == "ValidationError"
    assert json.loads(result.content)["status"] == "failed"
    work.assert_not_called()
    work.side_effect = RuntimeError("worker failed")
    result = await executor.execute_async(call(arguments={"query": "ok"}))
    assert result.is_error
    assert result.error == "worker failed"
    assert not json.loads(result.content)["error"]["retryable"]


@pytest.mark.asyncio
async def test_native_child_timeout_and_parent_todo_override_and_usage():
    seen = []
    async def generate(chat, **kwargs):
        seen.append((chat.messages, kwargs["tools"]))
        return response(tokens=7)
    executor = TaskExecutor(client=client_for(generate), nested_llm_nodes=True)
    executor.register_task(manifest(nested_tools=[], nested_system_prompt="Read-only research.", budget_cascade=True), AsyncMock())
    # Execute a native child through a parent tool to exercise the context boundary.
    parent = AsyncAgentLoop(client_for(), builtin_todo_tools=False)
    from magic_llm.agent._loop_shared import PARENT_TODO_TOOLS
    state = AgentState(start_time=time.monotonic())
    tokens = (PARENT_STATE.set(state), PARENT_BUDGET.set(AgentBudget()), PARENT_TODO_TOOLS.set(parent._builtin_todo_enabled))
    try:
        result = await executor.execute_async(call(arguments={"query": "evidence"}))
        assert not result.is_error
        assert state.total_input_tokens == 7 and state.total_output_tokens == 7
        assert seen[0][1] == []
        assert seen[0][0][0]["content"] == "Read-only research."
    finally:
        PARENT_STATE.reset(tokens[0]); PARENT_BUDGET.reset(tokens[1]); PARENT_TODO_TOOLS.reset(tokens[2])
    cancelled = asyncio.Event()
    async def stall(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    executor._client = client_for(stall)
    short = manifest(nested_tools=[]).model_copy(update={"timeout_seconds": .03})
    executor.register_task(short, AsyncMock())
    result = await executor.execute_async(call(arguments={"query": "evidence"}))
    assert result.is_error and json.loads(result.content)["status"] == "timeout"
    assert cancelled.is_set()
    assert not any(get_all_depths().values())


@pytest.mark.asyncio
async def test_task_timeout_includes_queue_and_cancellation_releases_slot():
    executor = TaskExecutor()
    started = asyncio.Event()
    async def stall(query):
        started.set()
        await asyncio.Event().wait()
    executor.register_task(manifest(max_concurrency=1).model_copy(update={"timeout_seconds": .04}), stall)
    first = asyncio.create_task(executor.execute_async(call(arguments={"query": "first"})))
    await started.wait()
    second = await executor.execute_async(call(arguments={"query": "second"}))
    assert json.loads(second.content)["status"] == "timeout"
    await first
    assert not executor._task_semaphores["work"].locked()


def test_cascade_retains_parent_caps_when_child_limit_is_none():
    budget = _compute_child_budget(manifest(budget_cascade=True, nested_budget=AgentBudget()),
        AgentBudget(max_input_tokens=100, max_output_tokens=50, wall_clock_timeout=10),
        AgentState(total_input_tokens=80, total_output_tokens=45, start_time=time.monotonic()))
    assert budget.max_input_tokens == 20
    assert budget.max_output_tokens == 5
    assert 0 < budget.wall_clock_timeout <= 10


@pytest.mark.asyncio
async def test_empty_budget_does_not_spend_another_provider_call():
    client = client_for()
    loop = AsyncAgentLoop(client, budget=AgentBudget(max_input_tokens=0))
    with pytest.raises(AgentBudgetExceeded):
        await loop.run("test")
    client.llm.async_generate.assert_not_called()


@pytest.mark.asyncio
async def test_usage_chunk_before_final_chunk_is_accounted():
    async def chunks(*args, **kwargs):
        yield ChatCompletionModel(id="chunk", model="fake", choices=[],
            usage=UsageModel(prompt_tokens=10, completion_tokens=3, total_tokens=13))
        yield ChatCompletionModel(id="chunk", model="fake", choices=[ChoiceModel(index=0,
            delta=DeltaModel(content="answer"), finish_reason="stop")])
    client = client_for()
    client.llm.async_stream_generate = chunks
    loop = AsyncAgentLoop(client)
    async for _ in loop.stream("test"):
        pass
    assert loop.state.total_input_tokens == 10
    assert loop.state.total_output_tokens == 3


@pytest.mark.parametrize("loop_type", [AsyncAgentLoop, AgentLoop])
def test_state_snapshot_cannot_mutate_nested_message_payload(loop_type):
    loop = loop_type(client_for())
    loop._state.messages = [{"role": "user", "content": [{"text": "original"}]}]
    snapshot = loop.state
    snapshot.messages[0]["content"][0]["text"] = "changed"
    assert loop.state.messages[0]["content"][0]["text"] == "original"


@pytest.mark.asyncio
async def test_thread_tools_keep_run_context():
    state = AgentState(step=8)
    token = PARENT_STATE.set(state)
    try:
        assert await run_sync_in_thread(lambda: PARENT_STATE.get().step) == 8
    finally:
        PARENT_STATE.reset(token)


def test_todo_rejection_does_not_partially_mutate_state():
    state = TodoState()
    item = {"id": 1, "content": "Read evidence", "status": "pending", "priority": "high"}
    state.replace([item])
    for invalid in [[{**item, "status": []}], [{**item, "unexpected": True}], [{**item, "content": "x" * 2001}], [item] * 101]:
        with pytest.raises(ValueError):
            state.replace(invalid)
        assert state.snapshot()["todos"] == [item]


@pytest.mark.asyncio
async def test_provider_stream_keeps_contextvars_across_chunks_with_deadline():
    import contextvars
    current = contextvars.ContextVar("provider", default=None)
    closed = []
    async def chunks(*args, **kwargs):
        token = current.set("live")
        try:
            yield ChatCompletionModel(id="chunk", model="fake", choices=[])
            assert current.get() == "live"
            yield ChatCompletionModel(id="chunk", model="fake", choices=[ChoiceModel(
                index=0, delta=DeltaModel(content="done"), finish_reason="stop")])
        finally:
            current.reset(token)
            closed.append(True)
    client = client_for()
    client.llm.async_stream_generate = chunks
    loop = AsyncAgentLoop(client, budget=AgentBudget(wall_clock_timeout=1))
    async for _ in loop.stream("test"):
        pass
    assert closed == [True]
    assert current.get() is None


@pytest.mark.asyncio
async def test_stream_budget_stops_before_next_chunk_and_preserves_partial_usage():
    continued = []
    closed = []
    async def chunks(*args, **kwargs):
        try:
            yield ChatCompletionModel(id="chunk", model="fake", choices=[],
                usage=UsageModel(prompt_tokens=10, completion_tokens=3, total_tokens=13))
            continued.append(True)
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    client = client_for()
    client.llm.async_stream_generate = chunks
    loop = AsyncAgentLoop(client, budget=AgentBudget(max_input_tokens=5))
    with pytest.raises(AgentBudgetExceeded):
        async for _ in loop.stream("test"):
            pass
    assert loop.state.total_input_tokens == 10
    assert not continued
    # Explicit source closing must complete before the public stream returns.
    assert closed == [True]


@pytest.mark.asyncio
async def test_raw_loop_tool_binding_cannot_remove_task_schema_safeguards():
    from magic_llm.agent._loop_shared import _register_tools_with_executor
    executor = TaskExecutor()
    work = AsyncMock(return_value="done")
    executor.register_task(manifest(), work)
    _register_tools_with_executor(executor, tool_functions={"work": work})
    result = await executor.execute_async(call(arguments={"query": 4}))
    assert result.is_error and result.error_type == "ValidationError"
    work.assert_not_called()


@pytest.mark.asyncio
async def test_forked_executors_keep_run_local_todos_separate():
    from magic_llm.agent._loop_shared import _register_tools_with_executor
    from magic_llm.agent.builtin_tools import create_builtin_todo_bundle
    original = TaskExecutor()
    work = AsyncMock(return_value="done")
    original.register_task(manifest(), work)
    first, second = original.fork(), original.fork()
    for executor in (first, second):
        _register_tools_with_executor(executor, builtin_tool_functions=create_builtin_todo_bundle()[1])
    await first.execute_async(call("todowrite", {"todos": [{"id": 1, "content": "private plan", "status": "pending", "priority": "high"}]}))
    result = await second.execute_async(call("todoread"))
    assert json.loads(result.content)["todos"] == []
    assert first._task_semaphores["work"] is second._task_semaphores["work"]
    assert "todoread" not in original._registry


@pytest.mark.parametrize("method", ["run_agent", "run_agent_stream", "run_agent_async", "run_agent_stream_async"])
def test_public_wrappers_forward_model_options_and_todo_setting(method, monkeypatch):
    from magic_llm import MagicLLM
    captured = {}
    class FakeLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self, **kwargs):
            if method.endswith("async"):
                async def answer():
                    return response()
                return answer()
            return response()
        def stream(self, **kwargs):
            if method.endswith("async"):
                async def answer():
                    if False:
                        yield
                return answer()
            return iter(())
    monkeypatch.setattr("magic_llm.agent.agent_loop.AgentLoop", FakeLoop)
    monkeypatch.setattr("magic_llm.agent.async_agent_loop.AsyncAgentLoop", FakeLoop)
    client = MagicLLM.__new__(MagicLLM)
    client._task_executor = None
    result = getattr(client, method)("test", temperature=.2, builtin_todo_tools=False)
    if "stream_async" in method:
        async def consume():
            async for _ in result:
                pass
        asyncio.run(consume())
    elif method.endswith("async"):
        asyncio.run(result)
    elif "stream" in method:
        list(result)
    assert captured["temperature"] == .2
    assert captured["builtin_todo_tools"] is False
