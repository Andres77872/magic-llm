# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
import json
import logging
import re
from typing import Callable, Type, Optional

from magic_llm.engine.base_chat import BaseChat
from magic_llm.engine.openai_adapters import (ProviderOpenAI,
                                              ProviderGroq,
                                              ProviderSambaNova,
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

logger = logging.getLogger(__name__)


class EngineOpenAI(BaseChat):
    engine = 'openai'

    # Map domain patterns to provider classes for easier maintenance
    PROVIDER_MAPPING = {
        r'api\.groq\.com': ProviderGroq,
        r'api\.sambanova\.ai': ProviderSambaNova,
        r'openrouter\.ai': ProviderOpenRouter,
        r'api\.mistral\.ai': ProviderMistral,
        r'api\.fireworks\.ai': ProviderFireworks,
        r'api\.deepseek\.com': ProviderDeepseek,
        r'api\.deepinfra\.com\/v1': ProviderDeepInfra,
    }

    def __init__(self,
                 api_key: str,
                 openai_adapter: Optional[Callable] = None,
                 base_url: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize the OpenAI engine with the appropriate provider.

        Args:
            api_key: The API key for authentication
            openai_adapter: Optional custom adapter class
            base_url: Optional base URL for the API
            **kwargs: Additional arguments for the provider
        """
        super().__init__(**kwargs)

        if openai_adapter is None and base_url is None:
            # Default to OpenAI if no adapter or base_url is provided
            self.base = ProviderOpenAI(api_key=api_key, **kwargs)
            logger.debug("Using default OpenAI provider")
        elif base_url:
            # Select provider based on base_url pattern matching
            provider_class = self._get_provider_for_url(base_url)
            self.base = provider_class(api_key=api_key, **kwargs)

            # If a custom base_url is provided for OpenAI, pass it along
            if provider_class == ProviderOpenAI:
                self.base = ProviderOpenAI(api_key=api_key, base_url=base_url, **kwargs)

            logger.debug(f"Selected provider {provider_class.__name__} for URL: {base_url}")
        elif isinstance(openai_adapter, type) and issubclass(openai_adapter, OpenAiBaseProvider):
            # Use the provided adapter class
            self.base = openai_adapter(api_key=api_key, **kwargs)
            logger.debug(f"Using custom adapter: {openai_adapter.__name__}")
        else:
            # Fallback to OpenAI
            self.base = ProviderOpenAI(api_key=api_key, **kwargs)
            logger.warning(f"Unrecognized adapter type: {type(openai_adapter)}. Using default OpenAI provider.")

    def _get_provider_for_url(self, url: str) -> Type[OpenAiBaseProvider]:
        """
        Determine the appropriate provider class based on the URL.

        Args:
            url: The base URL for the API

        Returns:
            The provider class to use
        """
        url_lower = url.lower()

        for pattern, provider in self.PROVIDER_MAPPING.items():
            if re.search(pattern, url_lower):
                return provider

        # Default to OpenAI if no match is found
        return ProviderOpenAI

    def prepare_response(self, r):
        return ModelChatResponse(**r)

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
                                        data=json.dumps(data),
                                        headers=self.base.headers)
            return response

    async def async_embedding(self, text: list[str] | str, **kwargs):
        async with AsyncHttpClient() as client:
            data = {
                "input": text,
                "model": self.model,
                **kwargs
            }
            response = await client.post_json(url=self.base.base_url + '/embeddings',
                                              data=json.dumps(data),
                                              headers=self.base.headers,
                                              timeout=kwargs.get('timeout'))
            return response

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        """
        Generate audio speech asynchronously.

        Args:
            data: The audio speech request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio speech response
        """
        return await self.base.async_audio_speech(data)

    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        """
        Generate audio speech synchronously.

        Args:
            data: The audio speech request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio speech response
        """
        return self.base.audio_speech(data) if hasattr(self.base, 'audio_speech') else None

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        """
        Generate audio transcriptions asynchronously.

        Args:
            data: The audio transcriptions request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio transcriptions response
        """
        return await self.base.async_audio_transcriptions(data)

    def sync_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        """
        Generate audio transcriptions synchronously.

        Args:
            data: The audio transcriptions request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio transcriptions response
        """
        return self.base.sync_audio_transcriptions(data)
