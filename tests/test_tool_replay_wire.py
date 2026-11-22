import copy
import json

import pytest

from magic_llm.engine.openai_adapters import ProviderOpenAI
from magic_llm.engine.openai_adapters.openai_deepinfra import ProviderDeepInfra
from magic_llm.engine.openai_adapters.openai_sambanova import ProviderSambaNova
from magic_llm.model import ModelChat


@pytest.mark.parametrize('provider',[ProviderOpenAI,ProviderDeepInfra,ProviderSambaNova])
def test_ui_tool_history_is_clean_on_wire_and_keeps_metadata_internally(provider):
    chat=ModelChat()
    call={'id':'call_1','type':'function','function':{'name':'lookup','arguments':'{"value":7}'},
          'index':0,'execution':'client','source':'node_tool','node_id':'tool-1'}
    chat.messages=[{'role':'user','content':'calculate'},
        {'role':'assistant','content':None,'tool_calls':[call]},
        {'role':'tool','tool_call_id':'call_1','content':'14','status':'success',
         'name':'lookup','execution_time_ms':4,'is_error':False}]
    original=copy.deepcopy(chat.messages)
    adapter=provider(api_key='test-key',model='test-model')
    body,_=adapter.prepare_data(chat)
    messages=json.loads(body)['messages']
    assert messages[1]['tool_calls']==[{k:call[k] for k in ('id','type','function')}]
    assert messages[2]=={'role':'tool','tool_call_id':'call_1','content':'14'}
    assert chat.messages==original
