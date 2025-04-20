import json

from tokenizers import Encoding

from magic_llm.model import ModelChat
from magic_llm.util import tokenizer

OPENAI_KEY = json.load(open('/home/andres/Documents/keys.json'))['openai']


def test_sync_openai_base_stream_generate_2():
    res: Encoding = tokenizer.from_hf('meta-llama/Llama-3.1-70B-Instruct', '''ML in Sales: Developed and maintained the “Celonis Quality Index” (CQI) model to predict likelihood of sales
conversion. Guided development of custom API to enable near-real-time predictions''')
    print(res)
    print(res)


def test_sync_openai_base_stream_generate_3():
    chat = ModelChat()
    chat.add_user_message('hi')
    print(chat.num_tokens_from_messages())


def test_sync_openai_base_stream_generate_4():
    chat = ModelChat('SYSTEM', max_input_tokens=500)
    chat.add_user_message('hi')
    chat.add_system_message('OTHER')

    for i in range(100):
        chat.add_user_message(f'MESSAGE USER {i}')
        chat.add_assistant_message(f'MESSAGE ASSISTANT {i}')

    chat.add_user_message('END')
    print(chat.get_messages())
