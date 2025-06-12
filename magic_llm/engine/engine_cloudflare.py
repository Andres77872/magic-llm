# https://developers.cloudflare.com/workers-ai/models/text-generation
import json
import time

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Message, Choice
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, ChoiceModel, DeltaModel
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineCloudFlare(BaseChat):
    engine = 'cloudflare'
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
            'Content-Type': 'application/json',
            **self.headers
        }
        data = {
            'messages': chat.get_messages(),
            **kwargs,
            **self.kwargs
        }
        json_data = json.dumps(data).encode('utf-8')
        return json_data, headers

    def process_generate(self, r: dict) -> ModelChatResponse:
        """Process Cloudflare response and convert to ModelChatResponse"""

        # Extract the result data
        result = r.get('result', {})

        # Create message
        message = Message(
            role='assistant',
            content=result.get('response', ''),
            tool_calls=None,
            refusal=None,
            annotations=[]
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            logprobs=None,
            finish_reason='stop' if r.get('success', True) else 'error'
        )

        # Create usage model
        usage_data = result.get('usage', {})
        usage = UsageModel(
            prompt_tokens=usage_data.get('prompt_tokens', 0),
            completion_tokens=usage_data.get('completion_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0)
        )

        # Create response
        return ModelChatResponse(
            id=f"cloudflare_{int(time.time() * 1000)}",
            object='chat.completion',
            created=int(time.time()),
            model='cloudflare',  # Cloudflare doesn't provide model name in response
            choices=[choice],
            usage=usage,
            service_tier=None,
            system_fingerprint=None
        )

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))
            return self.process_generate(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        data, headers = self.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.url,
                                        data=data,
                                        headers=headers)
            return self.process_generate(response)

    def prepare_stream_response(self, event):
        payload = json.loads(event[5:].strip())
        usage = None
        if 'usage' in payload:
            usage = UsageModel(
                prompt_tokens=payload['usage']['prompt_tokens'],
                completion_tokens=payload['usage']['completion_tokens'],
                total_tokens=payload['usage']['total_tokens'],
            )
        delta = DeltaModel(content=payload['response'], role=None)
        choice = ChoiceModel(delta=delta, finish_reason=None, index=0)
        return ChatCompletionModel(
            id='1',
            choices=[choice],
            created=int(time.time()),
            model=self.model,
            object='chat.completion.chunk',
            usage=usage or UsageModel(),
        )

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        data, headers = self.prepare_data(chat, stream=True, **kwargs)
        with HttpClient() as client:
            for event in client.stream_request("POST",
                                               url=self.url,
                                               data=data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if event != '\n' and event and event != 'data: [DONE]':
                    yield self.prepare_stream_response(event)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.prepare_data(chat, **kwargs, stream=True)

        async with AsyncHttpClient() as client:
            async for event in client.post_stream(
                    self.url,
                    data=json_data,
                    headers=headers,
                    timeout=kwargs.get('timeout')
            ):
                event = event.decode('utf-8').strip()
                if event != '\n' and event and event != 'data: [DONE]':
                    yield self.prepare_stream_response(event)
