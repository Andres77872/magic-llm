# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
import json
import urllib.request

import aiohttp

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class EngineOpenAI(BaseChat):
    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }

    def prepare_data(self, chat: ModelChat, **kwargs):
        # Construct the header and data to be sent in the request.

        data = {
            "model": self.model,
            "messages": chat.messages,
            **kwargs,
            **self.kwargs
        }

        if self.base_url == 'https://api.openai.com/v1':
            data.update({
                "stream_options": {
                    "include_usage": True
                }
            })

        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')
        return json_data, self.headers

    def prepare_http_data(self, chat: ModelChat, **kwargs):
        data, headers = self.prepare_data(chat, **kwargs)
        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url + '/chat/completions', data=data, headers=headers)

    def prepare_data_embedding(self, text: list[str] | str, **kwargs):
        # Construct the header and data to be sent in the request.

        data = {
            "input": text,
            "model": self.model,
            "encoding_format": "float",
            **kwargs
        }
        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url + '/embeddings', data=json_data, headers=self.headers)

    def prepare_response(self, r):
        if r['choices'][0]['message']['content']:
            return ModelChatResponse(**{
                'content': r['choices'][0]['message']['content'],
                'prompt_tokens': r['usage']['prompt_tokens'],
                'completion_tokens': r['usage']['completion_tokens'],
                'total_tokens': r['usage']['total_tokens'],
                'role': 'assistant'
            })
        else:  # interpret as function calling
            return ModelChatResponse(**{
                'content': r['choices'][0]['message']['tool_calls'][0]['function']['arguments'],
                'prompt_tokens': r['usage']['prompt_tokens'],
                'completion_tokens': r['usage']['completion_tokens'],
                'total_tokens': r['usage']['total_tokens'],
                'role': 'assistant'
            })

    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout'))

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url + '/chat/completions',
                                    data=json_data,
                                    headers=headers,
                                    timeout=timeout) as response:
                response_data = await response.read()
                encoding = response.charset or 'utf-8'

                r = json.loads(response_data.decode(encoding))
                return self.prepare_response(r)

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_http_data(chat, stream=False, **kwargs),
                                    timeout=kwargs.get('timeout')) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')

            # Decode and print the response.
            r = json.loads(response_data.decode(encoding))

            return self.prepare_response(r)

    def process_chunk(
            self, chunk: str,
            id_generation: str = '',
            last_chunk: ChatCompletionModel = None
    ) -> ChatCompletionModel:
        if chunk.startswith('data: ') and not chunk.endswith('[DONE]'):
            chunk = json.loads(chunk[5:])
            if self.base_url == 'https://api.groq.com/openai/v1':
                chunk['usage'] = chunk.get('x_groq', {}).get('usage')
            else:
                chunk['usage'] = c if (c := chunk.get('usage', {})) else {}
            if len(chunk['choices']) == 0:
                chunk['choices'] = [{}]
            chunk = ChatCompletionModel(**chunk)
            return chunk
        else:
            if c := chunk.strip():
                if c == 'data: [DONE]' and self.base_url == 'https://openrouter.ai/api/v1':
                    for i in range(3):
                        request = urllib.request.Request(f'https://openrouter.ai/api/v1/generation?id={id_generation}',
                                                         headers=self.headers)
                        with urllib.request.urlopen(request, timeout=2) as ses:
                            response = ses.read().decode('utf-8')
                            response = json.loads(response)
                            u = response['data']
                            usage = {
                                'completion_tokens': u['native_tokens_completion'],
                                'prompt_tokens': u['native_tokens_prompt']
                            }
                            last_chunk.usage = UsageModel(**usage)
                            last_chunk.choices[0].delta.content = ''
                            return last_chunk

    def stream_generate(self, chat: ModelChat, **kwargs):
        with urllib.request.urlopen(self.prepare_http_data(chat, stream=True, **kwargs),
                                    timeout=kwargs.get('timeout')) as response:
            id_generation = ''
            last_chunk = ''

            for chunk in response:
                chunk = chunk.decode('utf-8')
                if c := self.process_chunk(chunk.strip(), id_generation, last_chunk):
                    if c.id:
                        id_generation = c.id
                    last_chunk = c
                    yield c

    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout'))

        async with aiohttp.ClientSession() as sess:
            async with sess.post(self.base_url + '/chat/completions',
                                 data=json_data,
                                 headers=headers,
                                 timeout=timeout) as response:
                id_generation = ''
                last_chunk = ''
                async for chunk in response.content:
                    chunk = chunk.decode('utf-8')
                    if c := self.process_chunk(chunk.strip(), id_generation, last_chunk):
                        if c.id:
                            id_generation = c.id
                        last_chunk = c
                        yield c

    def embedding(self, text: list[str] | str, **kwargs):
        with urllib.request.urlopen(self.prepare_data_embedding(text, **kwargs),
                                    timeout=kwargs.get('timeout')) as response:
            data = response.read().decode('utf-8')
            return data
