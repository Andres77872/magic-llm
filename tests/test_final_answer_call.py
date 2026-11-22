"""The call at the iteration limit is an answer (``AgentBudget.final_answer_notice``)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import AgentBudget, AgentBudgetExceeded
from magic_llm.model.ModelChatResponse import Choice, FunctionCall, Message, ModelChatResponse, ToolCall, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
from magic_llm.model.ModelChatStream import FunctionCall as StreamFunctionCall
from magic_llm.model.ModelChatStream import ToolCall as StreamToolCall

NOTICE = "<run_limit>Answer now with what you have.</run_limit>"


def _response(content=None, tool_calls=None):
    return ModelChatResponse(
        id="r", object="chat.completion", created=0.0, model="m",
        choices=[Choice(index=0, message=Message(role="assistant", content=content, tool_calls=tool_calls),
                        finish_reason="tool_calls" if tool_calls else "stop")],
        usage=UsageModel(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _call(call_id):
    return ToolCall(id=call_id, function=FunctionCall(name="lookup", arguments='{"q": "x"}'))


def _stream(content=None, call_id=None):
    chunk = ChatCompletionModel(id="c", model="m", choices=[ChoiceModel(
        index=0, delta=DeltaModel(content=content), finish_reason="tool_calls" if call_id else "stop")])
    if call_id:
        chunk.choices[0].delta.tool_calls = [StreamToolCall(
            index=0, id=call_id, function=StreamFunctionCall(name="lookup", arguments='{"q": "x"}'))]
    return [chunk]


class Recorder:
    """Answers each call in turn and records what each call was sent."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def record(self, chat, kwargs):
        self.calls.append({"messages": [dict(m) for m in chat.messages], "tool_choice": kwargs.get("tool_choice")})
        return self.answers.pop(0)


def _client(recorder, *, stream=False, sync=False):
    client = MagicMock()
    client.llm = MagicMock()
    if sync:
        client.llm.generate = lambda chat, **kw: recorder.record(chat, kw)
    elif stream:
        def async_stream_generate(chat, **kw):
            chunks = recorder.record(chat, kw)

            async def gen():
                for chunk in chunks:
                    yield chunk
            return gen()
        client.llm.async_stream_generate = async_stream_generate
    else:
        async def async_generate(chat, **kw):
            return recorder.record(chat, kw)
        client.llm.async_generate = async_generate
    return client


runs = []


def lookup(q):
    runs.append(q)
    return {"found": q}


@pytest.fixture(autouse=True)
def _reset_runs():
    runs.clear()


BUDGET = dict(max_iterations=2, final_answer_notice=NOTICE)


def _assert_final_call(recorder):
    first, final = recorder.calls
    assert first["tool_choice"] == "auto"
    assert final["tool_choice"] == "none"
    assert final["messages"][-1] == {"role": "user", "content": NOTICE}
    assert all(m.get("content") != NOTICE for m in first["messages"])


def test_sync_run_ends_with_an_answer_at_the_limit():
    recorder = Recorder([_response(tool_calls=[_call("c1")]), _response(content="The answer.")])
    loop = AgentLoop(_client(recorder, sync=True), tools=[lookup], budget=AgentBudget(**BUDGET))
    response = loop.run("q?")
    assert response.content == "The answer."
    _assert_final_call(recorder)
    assert all(m.get("content") != NOTICE for m in loop.state.messages)


def test_async_run_ends_with_an_answer_at_the_limit():
    recorder = Recorder([_response(tool_calls=[_call("c1")]), _response(content="The answer.")])
    loop = AsyncAgentLoop(_client(recorder), tools=[lookup], budget=AgentBudget(**BUDGET))
    response = asyncio.run(loop.run("q?"))
    assert response.content == "The answer."
    _assert_final_call(recorder)


def test_async_stream_ends_with_an_answer_at_the_limit():
    recorder = Recorder([_stream(call_id="c1"), _stream(content="The answer.")])
    loop = AsyncAgentLoop(_client(recorder, stream=True), tools=[lookup], budget=AgentBudget(**BUDGET))

    async def collect():
        return [chunk async for chunk in loop.stream("q?")]

    chunks = asyncio.run(collect())
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "The answer." in text
    _assert_final_call(recorder)
    assert all(m.get("content") != NOTICE for m in loop.state.messages)


def test_tool_calls_on_the_final_call_are_dropped():
    recorder = Recorder([_stream(call_id="c1"), _stream(content="Done.", call_id="c2")])
    loop = AsyncAgentLoop(_client(recorder, stream=True), tools=[lookup], budget=AgentBudget(**BUDGET))

    async def collect():
        return [chunk async for chunk in loop.stream("q?")]

    asyncio.run(collect())
    assert runs == ["x"]  # c1 ran, c2 did not
    last = loop.state.messages[-1]
    assert last["role"] == "assistant" and last["content"] == "Done." and not last.get("tool_calls")


def test_a_final_call_without_text_is_still_a_budget_error():
    recorder = Recorder([_response(tool_calls=[_call("c1")]), _response(tool_calls=[_call("c2")])])
    hooks = MagicMock()
    loop = AsyncAgentLoop(_client(recorder), tools=[lookup], budget=AgentBudget(**BUDGET), hooks=hooks)
    with pytest.raises(AgentBudgetExceeded) as raised:
        asyncio.run(loop.run("q?"))
    assert raised.value.budget_type == "max_iterations"
    hooks.on_budget_exceeded.assert_called_once()


def test_without_a_notice_the_limit_is_unchanged():
    recorder = Recorder([_response(tool_calls=[_call("c1")]), _response(tool_calls=[_call("c2")])])
    loop = AsyncAgentLoop(_client(recorder), tools=[lookup], budget=AgentBudget(max_iterations=2))
    with pytest.raises(AgentBudgetExceeded):
        asyncio.run(loop.run("q?"))
    assert [call["tool_choice"] for call in recorder.calls] == ["auto", "auto"]
