# https://developers.cloudflare.com/workers-ai/models/text-generation

import json
import urllib.request
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineCloudFlare(BaseChat):
    def __init__(self,
                 api_key: str,
                 account_id: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model}'
        self.api_key = api_key

    def prepare_data(self, chat: ModelChat, **kwargs):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            'messages': chat.messages,
            'stream': self.stream,
            **kwargs
        }

        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.url, data=json_data, headers=headers, method='POST')

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        with urllib.request.urlopen(self.prepare_data(chat, **kwargs)) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')

            # Decode the response.
            r = json.loads(response_data.decode(encoding))

            r = r['result']['response']

            return ModelChatResponse(**{
                'content': r,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'role': 'assistant'
            })

    def stream_generate(self, chat: ModelChat, **kwargs):
        with urllib.request.urlopen(self.prepare_data(chat, **kwargs)) as response:
            for event in response:
                if event != b'\n':
                    event = event[5:].strip()
                    if event == b'[DONE]':
                        yield f'data: {chunk}\n'
                        yield f'\n'
                        return
                    event = json.loads(event.decode('utf-8'))
                    # print(event)
                    chunk = {
                        'id': '1',
                        'choices':
                            [{
                                'delta':
                                    {
                                        'content': event['response'],
                                        'role': None
                                    },
                                'finish_reason': None,
                                'index': 0
                            }],
                        'created': int(time.time()),
                        'model': self.model,
                        'object': 'chat.completion.chunk'
                    }
                    chunk = json.dumps(chunk)
                    yield f'data: {chunk}\n'
                    yield f'\n'
