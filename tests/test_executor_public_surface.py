"""The executor's public extension points, serial ordering, cancellation and fitting."""

import asyncio
import json

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.engine.tooling import map_request_tools


def call(name, call_id="c1", **arguments):
    return CanonicalToolCall(id=call_id, name=name, arguments=arguments)


class Refused(Exception):
    pass


class HostExecutor(ToolExecutor):
    """A host that reports its own refusals through the public hook."""

    @staticmethod
    def error_result(tool_call, start, exc):
        if not isinstance(exc, Refused):
            return ToolExecutor.error_result(tool_call, start, exc)
        return ToolResult(tool_call_id=tool_call.id, name=tool_call.name,
                          content=json.dumps({"refused": str(exc)}), is_error=True,
                          error=str(exc), error_type="refused")


def refuse():
    raise Refused("not in this workspace")


def boom():
    raise ValueError("bad")


@pytest.mark.parametrize("run", ["sync", "async"])
def test_error_result_is_the_override_point(run):
    executor = HostExecutor()
    executor.register("refuse", refuse)
    executor.register("boom", boom)
    if run == "sync":
        refused, failed = executor.execute(call("refuse")), executor.execute(call("boom"))
    else:
        refused = asyncio.run(executor.execute_async(call("refuse")))
        failed = asyncio.run(executor.execute_async(call("boom")))
    assert refused.error_type == "refused" and json.loads(refused.content) == {"refused": "not in this workspace"}
    assert failed.error_type == "ValueError"


def test_fingerprint_and_timeout_for_are_public():
    assert ToolExecutor.fingerprint("t", {"a": 1, "b": 2}) == ToolExecutor.fingerprint("t", {"b": 2, "a": 1})
    assert ToolExecutor.fingerprint("t", {"a": 1}) != ToolExecutor.fingerprint("u", {"a": 1})
    executor = ToolExecutor(per_tool_timeout=30, tool_timeouts={"slow": 300})
    assert executor.timeout_for("slow") == 300 and executor.timeout_for("other") == 30


def _concurrency_probe():
    state = {"now": 0, "max": 0}

    async def tool(n):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.02)
        state["now"] -= 1
        return n
    return state, tool


@pytest.mark.parametrize("serial,expected", [(True, 1), (False, 2)])
def test_concurrent_single_calls_of_a_serial_tool_are_ordered(serial, expected):
    state, tool = _concurrency_probe()
    executor = ToolExecutor(serial_tools={"edit"} if serial else None).fork()
    executor.register("edit", tool)

    async def two_at_once():
        return await asyncio.gather(executor.execute_async(call("edit", "a", n=1)),
                                    executor.execute_async(call("edit", "b", n=2)))

    results = asyncio.run(two_at_once())
    assert [r.content for r in results] == ["1", "2"]
    assert state["max"] == expected


def test_a_call_in_flight_can_be_cancelled_by_id():
    async def slow():
        await asyncio.sleep(10)
        return "late"

    executor = ToolExecutor(per_tool_timeout=30).fork()
    executor.register("slow", slow)

    async def scenario():
        task = asyncio.create_task(executor.execute_async(call("slow", "x1")))
        await asyncio.sleep(0.01)
        assert executor.running_call_ids() == {"x1"}
        assert executor.cancel("x1") is True
        result = await asyncio.wait_for(task, 1)
        return result

    result = asyncio.run(scenario())
    assert result.is_error and result.error_type == "ToolCallCancelled"
    assert executor.running_call_ids() == frozenset()
    assert executor.cancel("x1") is False


def test_cancelling_the_caller_still_propagates():
    async def slow():
        await asyncio.sleep(10)

    executor = ToolExecutor(per_tool_timeout=30).fork()
    executor.register("slow", slow)

    async def scenario():
        task = asyncio.create_task(executor.execute_async(call("slow", "x2")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert executor.running_call_ids() == frozenset()

    asyncio.run(scenario())


def big():
    return {"items": list(range(500)), "note": "complete"}


def test_a_result_over_its_limit_is_fitted_before_any_cut():
    seen = []

    def fitter(name, output, limit):
        seen.append((name, limit))
        return {"items": output["items"][:10], "note": "first 10 of 500"}

    executor = ToolExecutor(max_content_size=200, result_fitter=fitter)
    executor.register("big", big)
    result = asyncio.run(executor.execute_async(call("big")))
    assert seen == [("big", 200)]
    assert not result.is_error
    assert json.loads(result.content)["note"] == "first 10 of 500"


def test_a_fitter_that_declines_leaves_the_cut():
    executor = ToolExecutor(max_content_size=200)
    executor.set_result_fitter(lambda name, output, limit: None)
    executor.register("big", big)
    result = executor.execute(call("big"))
    assert result.is_error and result.error_type == "ContentTruncated"
    with pytest.raises(TypeError):
        executor.set_result_fitter("not callable")


def test_fork_has_its_own_run_state_and_shares_host_attributes():
    parent = HostExecutor(serial_tools={"edit"})
    parent.scope = object()
    child = parent.fork()
    assert child.scope is parent.scope
    assert child._running is not parent._running
    assert child._sync_serial_lock is not parent._sync_serial_lock


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_no_tools_means_no_tool_choice(provider):
    mapped = map_request_tools(provider, None, "auto")
    assert mapped.tools is None and mapped.tool_choice is None
    assert map_request_tools(provider, [], "required").tool_choice is None


def test_a_transport_stands_in_for_the_engine_and_keeps_its_dialect():
    from magic_llm import MagicLLM
    from magic_llm.engine.engine_anthropic import EngineAnthropic
    from magic_llm.engine.tooling import infer_provider_from_client
    from magic_llm.engine.transport import EngineTransport

    class Router(EngineTransport):
        routed = True

    client = MagicLLM(engine="anthropic", model="claude-x", private_key="k", retries=1,
                      transport=Router)
    assert isinstance(client.llm, Router)
    assert isinstance(client.llm.wrapped, EngineAnthropic)
    assert client.llm.retry_config.attempts == 1  # delegated to the engine
    assert client.llm.engine == "anthropic"
    assert infer_provider_from_client(client) == "anthropic"
    with pytest.raises(TypeError):
        MagicLLM(engine="openai", model="m", private_key="k", transport="router")
