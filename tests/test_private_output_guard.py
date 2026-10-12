"""Provider echoes are rejected before safe output reaches the caller."""
import json
import pickle
import pytest

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel
from magic_llm.util.private_limits import PrivateRequestLimits
from magic_llm.util.private_output import PrivateOutputGuard,PrivateOutputError

CANARY='FAKE_PRIVATE_OUTPUT_CREDENTIAL_CANARY'


@pytest.mark.parametrize('encoded',[False,True])
@pytest.mark.parametrize('chunk_size',[1,3,17])
def test_split_echo_is_withheld_even_inside_json_escapes(encoded,chunk_size):
    guard=PrivateOutputGuard(CANARY)
    secret=''.join('\\u%04x'%ord(c) for c in CANARY) if encoded else CANARY
    text='{"response":"'+secret+'"}'
    emitted=''
    with pytest.raises(PrivateOutputError):
        for i in range(0,len(text),chunk_size):
            emitted+=guard.feed(('tool',0),text[i:i+chunk_size])
    assert emitted=='{"response":"'
    assert CANARY not in repr(guard)
    guard.close()


@pytest.mark.parametrize('text',[
    'A quiet forest. FAKE_PRIVATE',
    '{"response":"safe \\"quote\\" and \\n text","value":"FAKE"}',
    '{"response":"A \\ud83d\\ude00 moon"}',
    'A literal unfinished \\u00',
    'A slash \\ and a path.',
])
def test_safe_fragmented_text_retains_exact_bytes_including_prefixes_and_unicode(text):
    guard=PrivateOutputGuard(CANARY)
    emitted=''.join(guard.feed('content',char) for char in text)+guard.finish('content')
    assert emitted==text
    guard.close()


def test_independent_tool_channels_do_not_mix_and_guard_objects_cannot_be_saved():
    guard=PrivateOutputGuard(CANARY)
    assert guard.feed('one','FAKE_')==''
    assert guard.feed('two','safe')=='safe'
    assert guard.finish('one')=='FAKE_'
    with pytest.raises(TypeError):pickle.dumps(guard)
    guard.close()
    assert guard._credential is None and guard._pending=={}
    with pytest.raises(PrivateOutputError):guard.feed('one','safe')


def test_metadata_and_hidden_provider_fields_are_checked():
    guard=PrivateOutputGuard(CANARY)
    chunk=ChatCompletionModel(id='safe',model='approved',choices=[{'delta':{'tool_calls':[
        {'index':0,'function':{'name':'edit','arguments':'{}'},'provider_metadata':{'hidden':CANARY}}
    ]}}])
    with pytest.raises(PrivateOutputError):guard.guard_chunk(chunk)
    guard.close()


class Provider:
    def __init__(self,location):
        self.location=location
        self.closed=False
    async def __aenter__(self):return self
    async def __aexit__(self,*args):self.closed=True
    async def post_json(self,**kwargs):
        message={'role':'assistant','content':'safe'}
        if self.location=='content':message['content']=CANARY
        elif self.location=='tool':message.update(content=None,tool_calls=[{
            'id':'call-1','type':'function','function':{'name':'edit','arguments':json.dumps({'response':CANARY})}}])
        result={'id':'safe','object':'chat.completion','created':0,'model':'approved',
            'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if self.location=='tool' else 'stop'}]}
        if self.location=='metadata':result['usage']={'provider_extra':{'trace':CANARY}}
        return result
    async def post_stream(self,url,**kwargs):
        if self.location=='metadata':
            yield ('data: '+json.dumps({'id':CANARY,'model':'approved','choices':[]})).encode()
            return
        text=CANARY if self.location=='content' else json.dumps({'response':CANARY})
        for index,char in enumerate(text):
            if self.location=='content':delta={'content':char}
            else:
                call={'index':0,'function':{'arguments':char}}
                if index==0:call.update(id='call-1',type='function');call['function']['name']='edit'
                delta={'tool_calls':[call]}
            yield ('data: '+json.dumps({'id':'safe','model':'approved','choices':[{'index':0,'delta':delta}]})).encode()
        yield ('data: '+json.dumps({'id':'safe','model':'approved','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('location',['content','tool','metadata'])
async def test_actual_private_adapter_rejects_echo_before_callback_or_caller(monkeypatch,caplog,stream,location):
    import importlib
    count=lambda text:list(range(len(str(text).encode('utf-8'))))
    monkeypatch.setattr('magic_llm.util.private_limits.from_openai',count)
    monkeypatch.setattr(importlib.import_module('magic_llm.model.ModelChat'),'from_openai',count)
    callbacks=[]
    async def callback(*args,**kwargs):callbacks.append(repr(args)+repr(kwargs))
    facade=MagicLLM(engine='openai',model='approved',private_key=CANARY,
        base_url='https://provider.example/v1',private_transport=True,retries=1,
        private_limits=PrivateRequestLimits(model='approved',max_input_tokens=1024,max_output_tokens=64,max_messages=4,timeout_seconds=5))
    facade.llm.callback=callback
    provider=Provider(location)
    monkeypatch.setattr(facade.llm,'_async_http_client',lambda:provider)
    chat=ModelChat();chat.add_user_message('A quiet forest.')
    emitted=[]
    try:
        with pytest.raises(Exception) as error:
            if stream:
                async for event in facade.llm.async_stream_generate(chat):emitted.append(event.model_dump_json())
            else:
                await facade.llm.async_generate(chat)
        assert provider.closed
        assert CANARY not in str(error.value)+''.join(emitted)+''.join(callbacks)+caplog.text
        assert not any('FAKE_PRIVATE_OUTPUT' in event for event in emitted)
    finally:
        facade.llm.executor.shutdown(wait=False)


def guarded_chunk(text=None, *, choice_index=0, call_index=0, tools=False, finish=None):
    delta = {'content': text} if text is not None else {}
    if tools:
        delta = {'tool_calls': [{'index': call_index, 'function': {'arguments': text}}]}
    return ChatCompletionModel(id='safe', model='approved', choices=[{
        'index': choice_index, 'delta': delta, 'finish_reason': finish}])


@pytest.mark.parametrize('tools', [False, True])
def test_guard_uses_the_runtime_identity_when_provider_indices_change(tools):
    guard = PrivateOutputGuard(CANARY)
    first = guard.guard_chunk(guarded_chunk(CANARY[:10], tools=tools))
    output = first.choices[0].delta
    assert (output.tool_calls[0].function.arguments if tools else output.content) == ''
    with pytest.raises(PrivateOutputError):
        guard.guard_chunk(guarded_chunk(CANARY[10:], tools=tools,
                                       choice_index=81, call_index=None))
    guard.close()


@pytest.mark.parametrize('tools', [False, True])
def test_finish_flushes_a_benign_prefix_on_the_correct_channel(tools):
    guard = PrivateOutputGuard(CANARY)
    first = guard.guard_chunk(guarded_chunk('Safe FAKE_', tools=tools, call_index=None))
    delta = first.choices[0].delta
    emitted = delta.tool_calls[0].function.arguments if tools else delta.content
    last = guard.guard_chunk(guarded_chunk(choice_index=27, finish='tool_calls' if tools else 'stop'))
    tail = last.choices[0].delta
    emitted += tail.tool_calls[0].function.arguments if tools else tail.content
    assert emitted == 'Safe FAKE_'
    if tools:
        assert tail.tool_calls[0].index == 0
    with pytest.raises(PrivateOutputError):
        guard.guard_chunk(guarded_chunk('late content', tools=tools))
    guard.close()


@pytest.mark.parametrize('layers',[1,2,3])
@pytest.mark.parametrize('chunk_size',[1,3,17])
def test_nested_json_escaped_credential_never_reaches_caller(layers,chunk_size):
    secret=''.join('\\u%04x'%ord(char) for char in CANARY)
    for _ in range(layers-1):
        secret=secret.replace('\\','\\\\')
    text='{"response":"'+secret+'"}'
    guard=PrivateOutputGuard(CANARY)
    emitted=''
    try:
        with pytest.raises(PrivateOutputError):
            for position in range(0,len(text),chunk_size):
                emitted+=guard.feed('arguments',text[position:position+chunk_size])
        assert emitted=='{"response":"'
        with pytest.raises(PrivateOutputError):guard.assert_safe({'metadata':text})
    finally:guard.close()


@pytest.mark.parametrize('text',[
    r'{"response":"safe \\u0041 text"}',
    r'A literal path \\u00 is unfinished.',
    r'A slash \\\\ and safe Unicode \\ud83d\\ude00.',
])
def test_safe_nested_escapes_preserve_original_bytes(text):
    guard=PrivateOutputGuard(CANARY)
    try:
        assert ''.join(guard.feed('content',char) for char in text)+guard.finish('content')==text
    finally:guard.close()
