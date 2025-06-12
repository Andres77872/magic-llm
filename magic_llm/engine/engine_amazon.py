import json

from magic_llm.engine.amazon_adapters import (
    ProviderAmazonNova,
    ProviderAmazonTitan,
    ProviderAmazonAnthropic,
    ProviderAmazonMeta
)
from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChatResponse, ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatStream import UsageModel


class EngineAmazon(BaseChat):
    engine = 'amazon'

    def __init__(self,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 **kwargs):
        super().__init__(**kwargs)

        # Determine which provider to use based on the model name
        if self.model.startswith('amazon.nova') or self.model.startswith('arn:aws:'):
            self.provider: AmazonBaseProvider = ProviderAmazonNova(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                service_name=service_name,
                **kwargs
            )
        elif self.model.startswith('amazon'):
            self.provider: AmazonBaseProvider = ProviderAmazonTitan(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                service_name=service_name,
                **kwargs
            )
        elif self.model.startswith('anthropic'):
            self.provider: AmazonBaseProvider = ProviderAmazonAnthropic(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                service_name=service_name,
                **kwargs
            )
        elif self.model.startswith('meta'):
            self.provider: AmazonBaseProvider = ProviderAmazonMeta(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                service_name=service_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        async with self.provider.aclient as client:
            response = await client.invoke_model(
                body=self.provider.prepare_data(chat, **kwargs),
                modelId=self.model,
                accept='application/json',
                contentType='application/json'
            )

            r = json.loads(await response['body'].read())
            return self.provider.process_response(r)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        response = self.provider.client.invoke_model(
            body=self.provider.prepare_data(chat, **kwargs),
            modelId=self.model,
            accept='application/json',
            contentType='application/json'
        )

        r = json.loads(response.get('body').read())
        return self.provider.process_response(r)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        response = self.provider.client.invoke_model_with_response_stream(
            body=self.provider.prepare_data(chat, **kwargs),
            modelId=self.model,
            accept='application/json',
            contentType='application/json'
        )

        for event in response.get("body"):
            event = json.loads(event["chunk"]["bytes"])
            chunk = self.provider.format_event_to_chunk(event)
            prompt_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('inputTokenCount', 0)
            completion_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('outputTokenCount', 0)
            chunk.usage = UsageModel(**{
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            })
            yield chunk

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        async with self.provider.aclient as client:
            response = await client.invoke_model_with_response_stream(
                body=self.provider.prepare_data(chat, **kwargs),
                modelId=self.model,
                accept='application/json',
                contentType='application/json'
            )

            async for event in response.get("body"):
                event = json.loads(event["chunk"]["bytes"])
                chunk = self.provider.format_event_to_chunk(event)
                prompt_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('inputTokenCount', 0)
                completion_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('outputTokenCount', 0)
                chunk.usage = UsageModel(**{
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                })
                yield chunk

    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        return self.provider.audio_speech(data, **kwargs)
