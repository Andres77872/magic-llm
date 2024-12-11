# https://docs.anthropic.com/claude/reference/messages-streaming
import aiohttp
import json
import urllib.request
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.http import AsyncHttpClient


class EngineAnthropic(BaseChat):
    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = 'https://api.anthropic.com/v1/messages'
        self.api_key = api_key

    def prepare_chunk(self, event: dict, idx, usage):
        chunk = None
        finish_reason = None
        if event['type'] == 'message_start':
            idx = event['message']['id']
            meta = event['message']['usage']
            usage = {
                "prompt_tokens": meta['input_tokens'] +
                                 meta.get('cache_read_input_tokens', 0) +
                                 meta.get('cache_creation_input_tokens', 0),
                "completion_tokens": meta['output_tokens'],
                "total_tokens": meta['input_tokens'] +
                                meta['output_tokens'] +
                                meta.get('cache_read_input_tokens', 0) +
                                meta.get('cache_creation_input_tokens', 0),
                "prompt_tokens_details": {'cached_tokens': meta['cache_read_input_tokens']}
            }
        if event['type'] == 'message_delta':
            finish_reason = event['delta']['stop_reason']
            meta = event['usage']
            usage['completion_tokens'] = meta['output_tokens']
            usage['total_tokens'] += meta['output_tokens']
        if event['type'] == 'content_block_delta':
            chunk = {
                'id': idx,
                'choices': [{
                    'delta': {
                        'content': event['delta']['text'],
                        'role': None
                    },
                    'finish_reason': finish_reason,
                    'index': 0
                }],
                'created': int(time.time()),
                'model': self.model,
                'usage': usage,
                'object': 'chat.completion.chunk'
            }
        return chunk, idx, usage

    def prepare_data(self, chat: ModelChat, **kwargs):
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'messages-2023-12-15,pdfs-2024-09-25, prompt-caching-2024-07-31',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }
        messages = chat.get_messages()
        preamble = messages[0]['content'] if messages[0]['role'] == 'system' else None
        if preamble:
            messages.pop(0)

        anthropic_chat = []
        for i in messages:
            if type(i['content']) is str:
                i['content'] = [{
                    "type": "text",
                    "text": i['content']
                }]
                t = i
            else:
                k = []
                for j in i['content']:
                    if j['type'] == 'text':
                        j.pop('image_url', None)
                        j.pop('document', None)
                        k.append(j)
                    elif j['type'] == 'image_url':
                        k.append({
                            'type': 'image',
                            'source': {
                                "type": "base64",
                                'media_type': j['image_url']['url'].split(',')[0].split(';')[0][5:],
                                'data': j['image_url']['url'].split(',')[1]
                            }
                        })
                    else:
                        k.append(j)
                i['content'] = k
                t = i
            anthropic_chat.append(t)
        ###
        result = []
        user_turns_processed = 0
        for turn in reversed(anthropic_chat):
            if turn["role"] == "user" and user_turns_processed < 3:
                t = []
                for i in turn["content"]:
                    if i["type"] == "text":
                        t.append({
                            "type": "text",
                            "text": i["text"],
                            "cache_control": {"type": "ephemeral"}
                        })
                    else:
                        t.append(i)
                result.append({
                    "role": "user",
                    "content": t
                })
                user_turns_processed += 1
            else:
                result.append(turn)
        anthropic_chat = list(reversed(result))
        ###
        data = {
            "model": self.model,
            "messages": anthropic_chat,
            **kwargs,
            **self.kwargs
        }
        if 'max_tokens' not in data:
            data['max_tokens'] = 4096

        if preamble:
            data['system'] = preamble

        json_data = json.dumps(data).encode('utf-8')
        return json_data, headers

    def prepare_http_data(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, **kwargs)
        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url, data=json_data, headers=headers)

    def process_chunk(self, chunk: str, idx, usage):
        chunk, idx, usage = self.prepare_chunk(json.loads(chunk), idx, usage)
        return ChatCompletionModel(**chunk) if chunk else None, idx, usage

    def process_generate(self, response):
        encoding = 'utf-8'
        r = json.loads(response.decode(encoding))
        return ModelChatResponse(**{
            'content': r['content'][0]['text'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=r['usage']['input_tokens'],
                completion_tokens=r['usage']['output_tokens'],
                total_tokens=r['usage']['input_tokens'] + r['usage']['output_tokens'],
            )
        })

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_raw_binary(url=self.base_url,
                                                    data=json_data,
                                                    headers=headers,
                                                    timeout=kwargs.get('timeout'))

            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        with urllib.request.urlopen(self.prepare_http_data(chat, stream=False, **kwargs)) as response:
            response_data = response.read()
            return self.process_generate(response_data)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_http_data(chat, stream=True, **kwargs)) as response:
            idx = None
            usage = None
            for chunk in response:
                if chunk:
                    evt = chunk.decode().split('data:')
                    if len(evt) != 2:
                        continue
                    if c := self.process_chunk(evt[-1].strip(), idx, usage):
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
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, data=json_data, headers=headers) as response:
                idx = None
                usage = None
                async for chunk in response.content:
                    if chunk:
                        evt = chunk.decode().split('data:')
                        if len(evt) != 2:
                            continue
                        if c := self.process_chunk(evt[-1].strip(), idx, usage):
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
