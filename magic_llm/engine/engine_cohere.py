# https://docs.cohere.com/reference/chat

import json
import urllib.request
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineCohere(BaseChat):
    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = 'https://api.cohere.ai/v1/chat'
        self.api_key = api_key

    def prepare_data(self, chat: ModelChat, **kwargs):
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }
        mp = {
            'user': 'User',
            'assistant': 'Chatbot'
        }

        preamble = None
        if chat.messages[0]['role'] == 'system':
            preamble = chat.messages.pop(0)['content']
        message = chat.messages.pop(-1)['content']
        for i in chat.messages:
            i['role'] = mp[i['role']]
        data = {
            "model": self.model,
            "messages": chat.messages,
            "message": message,
            "stream": self.stream,
            "preamble": preamble,
            "prompt_truncation": 'AUTO',
            **self.kwargs
        }

        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url, data=json_data, headers=headers)

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    def stram_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_data(chat, **kwargs)) as response:
            idx = None
            usage = None
            for chunk in response:
                if chunk:
                    event = json.loads(chunk)

                    if event['event_type'] == 'stream-start':
                        idx = event['generation_id']
                    elif event['event_type'] == 'stream-end':
                        meta = event['response']['meta']['billed_units']
                        usage = {
                            "prompt_tokens": meta['input_tokens'],
                            "completion_tokens": meta['output_tokens'],
                            "total_tokens": meta['input_tokens'] + meta['output_tokens']
                        }
                    else:
                        chunk = {
                            'id': idx,
                            'choices':
                                [{
                                    'delta':
                                        {
                                            'content': event['text'],
                                            'role': None
                                        },
                                    'finish_reason': None,
                                    'index': 0
                                }],
                            'created': int(time.time()),
                            'model': self.model,
                            'usage': usage,
                            'object': 'chat.completion.chunk'
                        }
                        chunk = json.dumps(chunk)
                        yield f'data: {chunk}\n'
                        yield f'\n'
            chunk = {
                'id': idx,
                'choices':
                    [{
                        'delta':
                            {
                                'content': '',
                                'role': None
                            },
                        'finish_reason': None,
                        'index': 0
                    }],
                'created': int(time.time()),
                'model': self.model,
                'usage': usage,
                'object': 'chat.completion.chunk'
            }
            chunk = json.dumps(chunk)
            yield f'data: {chunk}\n'
            yield f'\n'
            yield f'[DONE]'
            yield f'\n'
