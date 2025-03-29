# https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=es-419
import json
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineGoogle(BaseChat):
    engine = 'google'

    def __init__(self,
                 api_key: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        base = 'https://generativelanguage.googleapis.com/v1beta/models/'
        self.url_stream = f'{base}{self.model}:streamGenerateContent?alt=sse&key={api_key}'
        self.url = f'{base}{self.model}:generateContent?key={api_key}'
        self.api_key = api_key

    def prepare_data(self, chat: ModelChat, **kwargs):
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
            **self.headers
        }
        messages = chat.get_messages()

        if 'system' in messages[0]['role']:
            preamble = messages.pop(0)['content']
        else:
            preamble = None

        data = {
            "contents": [
                {
                    "role": x['role'].replace('assistant', 'model'),
                    "parts": [
                        # check if x['content'] is a list or a string
                        *(
                            [
                                {"text": part["text"]} if part["type"] == "text" else
                                {
                                    "inline_data": {
                                        "mime_type": part["image_url"]["url"].split(";")[0].replace('data:', ''),
                                        "data": part["image_url"]["url"].split(",")[1]
                                    }
                                }
                                for part in x['content']
                            ] if isinstance(x['content'], list)
                            else [{"text": x['content']}]
                        )
                    ]
                }
                for x in messages
            ],
            "generationConfig": {**self.kwargs},
            **kwargs,
        }
        if preamble is not None:
            data['systemInstruction'] = {
                "parts": [{'text': preamble}]
            }

        json_data = json.dumps(data).encode('utf-8')
        return json_data, headers, data

    def process_generate(self, data: dict):
        content = data['candidates'][0]['content']['parts'][0]['text']
        return ModelChatResponse(**{
            'content': content,
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=data['usageMetadata']['promptTokenCount'],
                completion_tokens=data['usageMetadata']['candidatesTokenCount'],
                total_tokens=data['usageMetadata']['totalTokenCount']
            )
        })

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))
            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers, data = self.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.url,
                                        data=json_data,
                                        headers=headers)
            return self.process_generate(response)

    def prepare_stream_response(self, chunk):
        chunk = chunk.strip()
        chunk = json.loads(chunk[5:].strip())
        chunk = {
            'id': '1',
            'choices': [{
                'delta': {
                    'content': chunk['candidates'][0]['content']['parts'][0]['text'],
                    'role': 'assistant'
                },
                'finish_reason': None,
                'index': 0
            }],
            'created': int(time.time()),
            'model': self.model,
            'object': 'chat.completion.chunk',
            'usage': {
                'prompt_tokens': chunk['usageMetadata']['promptTokenCount'],
                'completion_tokens': chunk['usageMetadata'].get('candidatesTokenCount', 0),
                'total_tokens': chunk['usageMetadata']['totalTokenCount']
            }
        }
        return ChatCompletionModel(**chunk)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers, data = self.prepare_data(chat, **kwargs)

        with HttpClient() as client:
            for chunk in client.stream_request("POST",
                                               self.url_stream,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if chunk.strip():
                    yield self.prepare_stream_response(chunk)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers, _ = self.prepare_data(chat, **kwargs)

        async with AsyncHttpClient() as client:
            async for chunk in client.post_stream(self.url_stream,
                                                  data=json_data,
                                                  headers=headers):
                if chunk.strip():
                    yield self.prepare_stream_response(chunk)
