"""A graph's invocation controls must run even when tool arguments repeat."""
import asyncio
import json

import pytest

from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL, ToolExecutor
from magic_llm.agent.types import CanonicalToolCall


def call(name, identifier):
    return CanonicalToolCall(id=identifier, name=name, arguments={"query": "same"})


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_controlled_calls_keep_identity_and_run_every_time_while_other_tools_cache(async_mode):
    observed = []
    ordinary_calls = []

    def controlled_sync(query):
        observed.append(CURRENT_TOOL_CALL.get().id)
        return {"query": query, "invocation": len(observed)}

    async def controlled_async(query):
        return controlled_sync(query)

    def ordinary_sync(query):
        ordinary_calls.append(query)
        return {"query": query}

    async def ordinary_async(query):
        return ordinary_sync(query)

    controlled = controlled_async if async_mode else controlled_sync
    controlled._disable_dedup = True
    executor = ToolExecutor(enable_dedup=True)
    executor.register("controlled", controlled)
    executor.register("ordinary", ordinary_async if async_mode else ordinary_sync)

    async def execute(item):
        return await executor.execute_async(item) if async_mode else executor.execute(item)

    results = [await execute(call("controlled", identifier)) for identifier in ["call-a", "call-b"]]
    assert observed == ["call-a", "call-b"]
    assert [result.tool_call_id for result in results] == observed
    assert [json.loads(result.content)["invocation"] for result in results] == [1, 2]
    assert all(not result.is_deduplicated for result in results)
    await execute(call("ordinary", "plain-a"))
    cached = await execute(call("ordinary", "plain-b"))
    assert ordinary_calls == ["same"]
    assert cached.is_deduplicated and cached.tool_call_id == "plain-b"
    assert CURRENT_TOOL_CALL.get() is None


@pytest.mark.asyncio
async def test_replacing_a_controlled_registration_restores_normal_cache_policy():
    async def controlled(query):
        return query
    controlled._disable_dedup = True
    hits = []
    async def ordinary(query):
        hits.append(query)
        return query
    executor = ToolExecutor(enable_dedup=True)
    executor.register("search", controlled)
    await executor.execute_async(call("search", "controlled"))
    executor.register("search", ordinary)
    await executor.execute_async(call("search", "plain-a"))
    result = await executor.execute_async(call("search", "plain-b"))
    assert hits == ["same"] and result.is_deduplicated
    executor.exclude_from_dedup("search")
    executor.register("search", controlled)
    executor.register("search", ordinary)
    await executor.execute_async(call("search", "excluded-a"))
    result = await executor.execute_async(call("search", "excluded-b"))
    assert hits == ["same", "same", "same"] and not result.is_deduplicated


@pytest.mark.asyncio
async def test_parallel_controlled_calls_are_independent_after_executor_fork():
    entered = []
    both_entered = asyncio.Event()

    async def controlled(query):
        entered.append(CURRENT_TOOL_CALL.get().id)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), 1)
        return CURRENT_TOOL_CALL.get().id

    controlled._disable_dedup = True
    executor = ToolExecutor(enable_dedup=True)
    executor.register("search", controlled)
    results = await executor.fork().execute_parallel_async([call("search", "left"), call("search", "right")])
    assert sorted(entered) == ["left", "right"]
    assert [json.loads(result.content) for result in results] == ["left", "right"]
    assert all(not result.is_error and not result.is_deduplicated for result in results)
