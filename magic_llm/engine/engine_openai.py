# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
from typing import Callable

from magic_llm.engine.base_chat import BaseChat
from magic_llm.engine.openai_adapters import (ProviderOpenAI,
                                              ProviderGroq,
                                              ProviderSambaNova,
                                              ProviderLepton,
                                              ProviderOpenRouter,
                                              ProviderMistral,
                                              ProviderFireworks,
                                              ProviderDeepseek,
                                              ProviderDeepInfra,
                                              OpenAiBaseProvider)
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.model.ModelChatStream import UsageModel
from magic_llm.util.http import AsyncHttpClient, HttpClient


class EngineOpenAI(BaseChat):
    engine = 'openai'
    def __init__(self,
                 api_key: str,
                 openai_adapter: Callable = None,
                 base_url: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if openai_adapter is None and base_url is None:
            self.base: OpenAiBaseProvider = ProviderOpenAI(api_key=api_key, **kwargs)
        elif base_url:
            if 'api.groq.com' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderGroq(api_key=api_key, **kwargs)
            elif 'api.sambanova.ai' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderSambaNova(api_key=api_key, **kwargs)
            elif 'lepton.run' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderLepton(api_key=api_key, **kwargs)
            elif 'openrouter.ai' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderOpenRouter(api_key=api_key, **kwargs)
            elif 'api.mistral.ai' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderMistral(api_key=api_key, **kwargs)
            elif 'api.fireworks.ai' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderFireworks(api_key=api_key, **kwargs)
            elif 'api.deepseek.com' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderDeepseek(api_key=api_key, **kwargs)
            elif 'api.deepinfra.com/v1' in base_url.lower():
                self.base: OpenAiBaseProvider = ProviderDeepInfra(api_key=api_key, **kwargs)
            else:
                self.base: OpenAiBaseProvider = ProviderOpenAI(api_key=api_key, base_url=base_url, **kwargs)
        elif type(openai_adapter) is OpenAiBaseProvider:
            self.base: OpenAiBaseProvider = openai_adapter(api_key=api_key, **kwargs)

    def prepare_response(self, r):
        if r['choices'][0]['message'].get('content'):
            return ModelChatResponse(**{
                'content': r['choices'][0]['message']['content'],
                'role': 'assistant',
                'usage': UsageModel(
                    prompt_tokens=r['usage']['prompt_tokens'],
                    completion_tokens=r['usage']['completion_tokens'],
                    total_tokens=r['usage']['total_tokens']
                )
            })
        else:  # interpret as function calling
            return ModelChatResponse(**{
                'content': r['choices'][0]['message']['tool_calls'][0]['function']['arguments'],
                'role': 'assistant',
                'usage': UsageModel(
                    prompt_tokens=r['usage']['prompt_tokens'],
                    completion_tokens=r['usage']['completion_tokens'],
                    total_tokens=r['usage']['total_tokens']
                )
            })

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self.base.prepare_data(chat, **kwargs)
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=self.base.base_url + '/chat/completions',
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout'))
            return self.prepare_response(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Make the request and read the response.
        data, headers = self.base.prepare_data(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.base.base_url + '/chat/completions',
                                        data=data,
                                        headers=headers)
            return self.prepare_response(response)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        data, headers = self.base.prepare_data(chat, stream=True, **kwargs)
        with HttpClient() as client:
            id_generation = ''
            last_chunk = ''
            for chunk in client.stream_request("POST",
                                               self.base.base_url + '/chat/completions',
                                               data=data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout')):
                if c := self.base.process_chunk(chunk.strip(), id_generation, last_chunk):
                    if c.id:
                        id_generation = c.id
                    last_chunk = c
                    yield c

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self.base.prepare_data(chat, stream=True, **kwargs)
        async with AsyncHttpClient() as client:
            id_generation = ''
            last_chunk = ''
            async for chunk in client.post_stream(self.base.base_url + '/chat/completions',
                                                  data=json_data,
                                                  headers=headers,
                                                  timeout=kwargs.get('timeout')):
                chunk = chunk.decode('utf-8')
                if c := self.base.process_chunk(chunk.strip(), id_generation, last_chunk):
                    if c.id:
                        id_generation = c.id
                    last_chunk = c
                    yield c

    def embedding(self, text: list[str] | str, **kwargs):
        with HttpClient() as client:
            data = {
                "input": text,
                "model": self.model,
                **kwargs
            }
            response = client.post_json(url=self.base.base_url + '/embeddings',
                                        data=data,
                                        headers=self.base.headers)
            return response

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        return await self.base.async_audio_speech(data)

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        return await self.base.async_audio_transcriptions(data)
