# https://docs.cohere.com/reference/chat
import json
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Message, Choice
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, ChoiceModel, DeltaModel
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineCohere(BaseChat):
    engine = 'cohere'

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
        messages = chat.get_messages().copy()
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

    def process_generate(self, r: dict) -> ModelChatResponse:
        """Process Cohere response and convert to ModelChatResponse"""

        # Map Cohere finish reasons to OpenAI format
        finish_reason_map = {
            'COMPLETE': 'stop',
            'MAX_TOKENS': 'length',
            'ERROR': 'stop',
            'ERROR_TOXIC': 'content_filter',
            'ERROR_LIMIT': 'length',
            'USER_CANCEL': 'stop'
        }
        finish_reason = finish_reason_map.get(
            r.get('finish_reason', 'COMPLETE'),
            'stop'
        )

        # Create message
        message = Message(
            role='assistant',
            content=r.get('text', ''),
            tool_calls=None,  # Cohere tool calls would need separate handling
            refusal=None,
            annotations=[]
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            logprobs=None,  # Cohere doesn't provide logprobs in this response
            finish_reason=finish_reason
        )

        # Create usage model
        tokens = r['meta']['tokens']
        usage = UsageModel(
            prompt_tokens=tokens['input_tokens'],
            completion_tokens=tokens['output_tokens'],
            total_tokens=tokens['input_tokens'] + tokens['output_tokens']
        )

        # Create response
        return ModelChatResponse(
            id=r.get('response_id', r.get('generation_id', f"cohere_{int(time.time() * 1000)}")),
            object='chat.completion',
            created=int(time.time()),
            model='cohere',  # Cohere doesn't provide model name in response
            choices=[choice],
            usage=usage,
            service_tier=None,  # Cohere doesn't provide this
            system_fingerprint=None  # Cohere doesn't provide this
        )

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.base_url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))
            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.base_url,
                                        data=json_data,
                                        headers=headers)
            return self.process_generate(response)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, stream=True, **kwargs)
        with HttpClient() as client:
            idx = None
            usage = None
            for chunk in client.stream_request("POST",
                                               self.base_url,
                                               data=json_data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if chunk:
                    if c := self.process_chunk(chunk.strip(), idx, usage):
                        idx = c[1]
                        usage = c[2]
                        if c[0]:
                            yield c[0]
            delta = DeltaModel(content='', role=None)
            choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
            yield ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage or UsageModel(),
            )

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, **kwargs, stream=True)

        async with AsyncHttpClient() as client:
            idx = None
            usage = None
            async for chunk in client.post_stream(self.base_url,
                                                  data=json_data,
                                                  headers=headers):
                if chunk:
                    if c := self.process_chunk(chunk.decode().strip(), idx, usage):
                        idx = c[1]
                        usage = c[2]
                        if c[0]:
                            yield c[0]

            # Final chunk after the stream is complete
            delta = DeltaModel(content='', role=None)
            choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
            yield ChatCompletionModel(
                id=idx,
                choices=[choice],
                created=int(time.time()),
                model=self.model,
                object='chat.completion.chunk',
                usage=usage or UsageModel(),
            )

    def process_chunk(self, chunk: str, idx, usage):
        model, idx, usage = self.prepare_chunk(json.loads(chunk), idx, usage)
        return model, idx, usage

    def prepare_chunk(self, event: dict, idx, usage):
        if event['event_type'] == 'stream-start':
            idx = event['generation_id']
            return None, idx, usage
        if event['event_type'] == 'stream-end':
            meta = event['response']['meta']['billed_units']
            usage = UsageModel(
                prompt_tokens=meta['input_tokens'],
                completion_tokens=meta['output_tokens'],
                total_tokens=meta['input_tokens'] + meta['output_tokens'],
            )
            return None, idx, usage

        delta = DeltaModel(content=event['text'], role=None)
        choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
        model = ChatCompletionModel(
            id=idx,
            choices=[choice],
            created=int(time.time()),
            model=self.model,
            object='chat.completion.chunk',
            usage=usage or UsageModel(),
        )
        return model, idx, usage
