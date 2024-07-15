# https://docs.cohere.com/reference/chat
import aiohttp
import json
import urllib.request
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel


class EngineCohere(BaseChat):
    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = 'https://api.cohere.ai/v1/chat'
        self.api_key = api_key

    def prepare_data(self, chat: ModelChat, **kwargs):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }
        messages = chat.get_messages()
        mp = {'user': 'User', 'assistant': 'Chatbot'}
        if 'system' in messages[0]['role']:
            preamble = messages.pop(0)['content']
        else:
            preamble = None
        message = messages.pop(-1)['content']
        for i in messages:
            i['role'] = mp[i['role']]
        data = {
            "model": self.model,
            "messages": messages,
            "message": message,
            "preamble": preamble,
            "prompt_truncation": 'AUTO',
            **kwargs
        }
        json_data = json.dumps(data).encode('utf-8')
        return json_data, headers

    def prepare_http_data(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, **kwargs)
        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url, data=json_data, headers=headers)

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout'))

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url,
                                    data=json_data,
                                    headers=headers,
                                    timeout=timeout) as response:
                response_data = await response.read()
                encoding = response.charset or 'utf-8'
                r = json.loads(response_data.decode(encoding))

                return ModelChatResponse(**{
                    'content': r['text'],
                    'prompt_tokens': r['meta']['tokens']['input_tokens'],
                    'completion_tokens': r['meta']['tokens']['input_tokens'] + r['meta']['tokens']['output_tokens'],
                    'total_tokens': r['meta']['tokens']['output_tokens'],
                    'role': 'assistant'
                })

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_http_data(chat, **kwargs)) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')
            r = json.loads(response_data.decode(encoding))

            return ModelChatResponse(**{
                'content': r['text'],
                'prompt_tokens': r['meta']['tokens']['input_tokens'],
                'completion_tokens': r['meta']['tokens']['input_tokens'] + r['meta']['tokens']['output_tokens'],
                'total_tokens': r['meta']['tokens']['output_tokens'],
                'role': 'assistant'
            })

    def stream_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_http_data(chat, **kwargs, stream=True)) as response:
            idx = None
            usage = None
            for chunk in response:
                if chunk:
                    if c := self.process_chunk(chunk.strip(), idx, usage):
                        idx = c[1]
                        usage = c[2]
                        if c[0]:
                            yield c[0]
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
            yield ChatCompletionModel(**chunk)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, **kwargs, stream=True)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, data=json_data, headers=headers) as response:
                idx = None
                usage = None
                async for chunk in response.content:
                    if chunk:
                        if c := self.process_chunk(chunk.decode().strip(), idx, usage):
                            idx = c[1]
                            usage = c[2]
                            if c[0]:
                                yield c[0]
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
                yield ChatCompletionModel(**chunk)

    def process_chunk(self, chunk: str, idx, usage):
        chunk, idx, usage = self.prepare_chunk(json.loads(chunk), idx, usage)
        return ChatCompletionModel(**chunk) if chunk else None, idx, usage

    def prepare_chunk(self, event: dict, idx, usage):
        chunk_data = None
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
            chunk_data = {
                'id': idx,
                'choices': [{
                    'delta': {
                        'content': event['text'],
                        'role': None
                    },
                    'finish_reason': None,
                    'index': 0
                }],
                'created': int(time.time()),
                'model': self.model,
                'object': 'chat.completion.chunk'
            }
            if usage:
                chunk_data.update({
                    'usage': usage
                })

        return chunk_data, idx, usage
