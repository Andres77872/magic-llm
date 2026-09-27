"""Ambiguous IDs must fail before dispatch, in all four loop modes."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from magic_llm.agent.agent_loop import AgentLoop
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent._loop_shared import PARENT_STATE
from magic_llm.model.ModelChat import ModelChat
from magic_llm.model.ModelChatResponse import Choice,Message,ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel,ChoiceModel,DeltaModel


def call(identifier,value):
    return {'id':identifier,'type':'function','function':{'name':'work','arguments':json.dumps({'value':value})}}


def response(calls=None):
    return ModelChatResponse(id='response',object='chat.completion',created=0,model='fake',
        choices=[Choice(index=0,message=Message(role='assistant',content='' if calls else 'done',tool_calls=calls),
            finish_reason='tool_calls' if calls else 'stop')])


class Engine:
    def __init__(self,responses): self.responses=iter(responses);self.count=0
    def generate(self,*args,**kwargs): self.count+=1;return next(self.responses)
    async def async_generate(self,*args,**kwargs): return self.generate(*args,**kwargs)
    def stream_generate(self,*args,**kwargs):
        result=self.generate(*args,**kwargs)
        calls=[c.model_dump() for c in result.tool_calls or []]
        for index,item in enumerate(calls):item['index']=index
        yield ChatCompletionModel(id='response',model='fake',choices=[ChoiceModel(index=0,delta=DeltaModel(tool_calls=calls or None),finish_reason=result.finish_reason)])
    async def async_stream_generate(self,*args,**kwargs):
        for chunk in self.stream_generate(*args,**kwargs):yield chunk


@pytest.mark.parametrize('mode',['sync','sync_stream','async','async_stream'])
@pytest.mark.parametrize('case',['batch','round','history','blank'])
def test_repeated_or_blank_ids_never_execute_ambiguous_calls(mode,case):
    executed=[]
    def work(value): executed.append(value);return value*2
    initial=None
    if case=='batch':steps=[response([call('same',2),call('same',7)])];expected=[]
    elif case=='round':steps=[response([call('same',2)]),response([call('same',7)])];expected=[2]
    elif case=='history':
        initial=ModelChat();initial.messages=[{'role':'user','content':'before'},
            {'role':'assistant','content':None,'tool_calls':[call('same',1)]},
            {'role':'tool','tool_call_id':'same','content':'2'}]
        steps=[response([call('same',7)])];expected=[]
    else:steps=[response([call('',7)])];expected=[]
    engine=Engine(steps)
    client=SimpleNamespace(llm=engine)
    loop_class=AsyncAgentLoop if mode.startswith('async') else AgentLoop
    loop=loop_class(client,tools=[work],builtin_todo_tools=False)
    parent=PARENT_STATE.get()
    with pytest.raises(ValueError,match='tool call ID|Tool call ID'):
        if mode=='sync':loop.run('compute',initial_chat=initial)
        elif mode=='sync_stream':list(loop.stream('compute',initial_chat=initial))
        elif mode=='async':asyncio.run(loop.run('compute',initial_chat=initial))
        else:
            async def consume():
                async for _ in loop.stream('compute',initial_chat=initial):pass
            asyncio.run(consume())
    assert executed==expected
    assert PARENT_STATE.get() is parent


@pytest.mark.parametrize('asynchronous',[False,True])
def test_duplicate_replay_ids_fail_before_provider_and_do_not_leak_context(asynchronous):
    chat=ModelChat()
    chat.messages=[{'role':'assistant','tool_calls':[call('same',2),call('same',7)],'content':None}]
    engine=Engine([])
    loop=(AsyncAgentLoop if asynchronous else AgentLoop)(SimpleNamespace(llm=engine),builtin_todo_tools=False)
    parent=PARENT_STATE.get()
    with pytest.raises(ValueError,match='Duplicate tool call ID'):
        if asynchronous:asyncio.run(loop.run('',initial_chat=chat))
        else:loop.run('',initial_chat=chat)
    assert engine.count==0
    assert PARENT_STATE.get() is parent
