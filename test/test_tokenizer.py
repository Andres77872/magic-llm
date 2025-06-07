import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tokenizers import Encoding

from magic_llm.util import tokenizer

OPENAI_KEY = json.load(open('/home/andres/Documents/keys.json'))['openai']


def test_sync_openai_base_stream_generate_2():
    res: Encoding = tokenizer.from_hf('meta-llama/Llama-3.1-70B-Instruct', '''ML in Sales: Developed and maintained the “Celonis Quality Index” (CQI) model to predict likelihood of sales
conversion. Guided development of custom API to enable near-real-time predictions''')
    print(res)
    print(res)
