"""Tests for public run_agent* wrapper default budgets."""

import asyncio

from magic_llm import MagicLLM
from magic_llm.agent.types import AgentBudget
from magic_llm.model import ModelChatResponse


def _client():
    client = MagicLLM.__new__(MagicLLM)
    client._task_executor = None
    return client


def test_run_agent_default_budget_is_150(monkeypatch):
    captured = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return ModelChatResponse(id="test", object="chat.completion", created=0, model="test", choices=[])

    monkeypatch.setattr("magic_llm.agent.agent_loop.AgentLoop", FakeAgentLoop)

    _client().run_agent("hello")

    assert captured["budget"].max_iterations == 150


def test_run_agent_stream_default_budget_is_150(monkeypatch):
    captured = {}

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stream(self, **kwargs):
            return iter(())

    monkeypatch.setattr("magic_llm.agent.agent_loop.AgentLoop", FakeAgentLoop)

    list(_client().run_agent_stream("hello"))

    assert captured["budget"].max_iterations == 150


def test_run_agent_async_default_budget_is_150(monkeypatch):
    captured = {}

    class FakeAsyncAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self, **kwargs):
            return ModelChatResponse(id="test", object="chat.completion", created=0, model="test", choices=[])

    monkeypatch.setattr("magic_llm.agent.async_agent_loop.AsyncAgentLoop", FakeAsyncAgentLoop)

    async def run_and_check():
        await _client().run_agent_async("hello")

    asyncio.run(run_and_check())

    assert captured["budget"].max_iterations == 150


def test_run_agent_stream_async_default_budget_is_150(monkeypatch):
    captured = {}

    class FakeAsyncAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def stream(self, **kwargs):
            if False:
                yield None

    monkeypatch.setattr("magic_llm.agent.async_agent_loop.AsyncAgentLoop", FakeAsyncAgentLoop)

    async def run_and_check():
        async for _ in _client().run_agent_stream_async("hello"):
            pass

    asyncio.run(run_and_check())

    assert captured["budget"].max_iterations == 150


def test_run_agent_explicit_budget_still_wins(monkeypatch):
    captured = {}
    custom_budget = AgentBudget(max_iterations=7)

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return ModelChatResponse(id="test", object="chat.completion", created=0, model="test", choices=[])

    monkeypatch.setattr("magic_llm.agent.agent_loop.AgentLoop", FakeAgentLoop)

    _client().run_agent("hello", budget=custom_budget)

    assert captured["budget"] is custom_budget
    assert captured["budget"].max_iterations == 7
