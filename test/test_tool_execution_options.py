"""Public execution options, call correlation and replay must stay local."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from magic_llm import MagicLLM
from magic_llm.agent._loop_shared import _build_initial_chat
from magic_llm.agent.task_executor import TaskExecutor
from magic_llm.agent.tool_executor import ToolExecutor, CURRENT_TOOL_CALL
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.model import ModelChat


@pytest.mark.asyncio
@pytest.mark.parametrize('synchronous',[False,True])
async def test_parallel_call_context_is_unique_and_does_not_escape(synchronous):
    async def async_tool(value):
        before=CURRENT_TOOL_CALL.get().id
        await asyncio.sleep(.003)
        return [before,CURRENT_TOOL_CALL.get().id,value]
    def sync_tool(value):
        return [CURRENT_TOOL_CALL.get().id,CURRENT_TOOL_CALL.get().id,value]
    executor=ToolExecutor()
    executor.register('work',sync_tool if synchronous else async_tool)
    calls=[CanonicalToolCall(id=str(i),name='work',arguments={'value':i}) for i in range(12)]
    results=await executor.execute_parallel_async(calls)
    assert [json.loads(r.content) for r in results]==[[str(i),str(i),i] for i in range(12)]
    assert CURRENT_TOOL_CALL.get() is None


def test_sync_execution_sets_call_context_and_resets_on_error():
    executor=ToolExecutor()
    def work():
        assert CURRENT_TOOL_CALL.get().id=='sync'
        raise RuntimeError('controlled')
    executor.register('work',work)
    result=executor.execute(CanonicalToolCall(id='sync',name='work',arguments={}))
    assert result.error_type=='RuntimeError'
    assert CURRENT_TOOL_CALL.get() is None


def test_limit_overrides_keep_task_executor_type_and_do_not_mutate_original():
    original=TaskExecutor(per_tool_timeout=15,max_parallel_tools=4)
    original.register('work',lambda:1)
    fork=original.with_options({'per_tool_timeout':.1,'max_parallel_tools':1,'max_content_size':25,'enable_dedup':True})
    assert isinstance(fork,TaskExecutor)
    assert fork._registry==original._registry
    assert (fork._per_tool_timeout,fork._max_parallel_tools,fork._max_content_size,fork._enable_dedup)==(.1,1,25,True)
    assert (original._per_tool_timeout,original._max_parallel_tools,original._max_content_size,original._enable_dedup)==(15,4,50000,False)


@pytest.mark.parametrize('options',[
    {'per_tool_timeout':0},{'per_tool_timeout':float('nan')},{'max_parallel_tools':1.5},
    {'max_parallel_tools':True},{'max_content_size':-1},{'enable_dedup':'false'},{'unknown':3},
])
def test_invalid_executor_options_fail_before_tools_run(options):
    with pytest.raises(ValueError): ToolExecutor().with_options(options)


def test_extra_messages_preserve_tool_metadata_and_are_copied():
    history=[{'role':'assistant','content':None,'tool_calls':[{'id':'old','type':'function',
        'function':{'name':'work','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'old','content':'6','name':'work'}]
    chat=_build_initial_chat(user_input='continue',extra_messages=history)
    assert chat.messages[:2]==history
    chat.messages[0]['tool_calls'][0]['id']='changed'
    assert history[0]['tool_calls'][0]['id']=='old'


@pytest.mark.parametrize('name', ['run_agent','run_agent_stream','run_agent_async','run_agent_stream_async'])
def test_all_public_wrappers_accept_initial_chat_and_options_without_provider_passthrough(monkeypatch,name):
    captured={}
    class SyncLoop:
        def __init__(self,**kwargs): captured['options']=kwargs
        def run(self,**kwargs): captured['run']=kwargs; return 'done'
        def stream(self,**kwargs): captured['run']=kwargs; return iter(())
    class AsyncLoop:
        def __init__(self,**kwargs): captured['options']=kwargs
        async def run(self,**kwargs): captured['run']=kwargs; return 'done'
        async def stream(self,**kwargs):
            captured['run']=kwargs
            if False: yield None
    monkeypatch.setattr('magic_llm.agent.agent_loop.AgentLoop',SyncLoop)
    monkeypatch.setattr('magic_llm.agent.async_agent_loop.AsyncAgentLoop',AsyncLoop)
    client=MagicLLM.__new__(MagicLLM)
    client._task_executor=None
    chat=ModelChat()
    chat.messages=[{'role':'tool','tool_call_id':'old','content':'6'}]
    opts={'max_parallel_tools':2}
    result=getattr(client,name)('',initial_chat=chat,tool_executor_options=opts)
    if name=='run_agent_async': asyncio.run(result)
    elif name=='run_agent_stream_async':
        async def consume():
            async for _ in result: pass
        asyncio.run(consume())
    elif name=='run_agent_stream': list(result)
    assert captured['options']['tool_executor_options']==opts
    assert captured['run']['initial_chat'] is chat


@pytest.mark.asyncio
async def test_closing_public_stream_closes_provider_before_returning():
    from magic_llm.agent._loop_shared import PARENT_STATE
    from magic_llm.model.ModelChatStream import ChatCompletionModel,ChoiceModel,DeltaModel
    closed=asyncio.Event()
    async def chunks(*args,**kwargs):
        try:
            yield ChatCompletionModel(id='partial',model='fake',choices=[ChoiceModel(index=0,delta=DeltaModel(content='partial'))])
            await asyncio.Event().wait()
        finally:
            closed.set()
    client=MagicLLM.__new__(MagicLLM)
    client._task_executor=None
    client.llm=SimpleNamespace(async_stream_generate=chunks)
    original=PARENT_STATE.get()
    stream=client.run_agent_stream_async('hello')
    await anext(stream)
    await stream.aclose()
    assert closed.is_set()
    assert PARENT_STATE.get() is original
