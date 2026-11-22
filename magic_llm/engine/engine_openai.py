# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models
import json
import logging
import math
from contextlib import closing, nullcontext
from typing import Callable, Type, Optional, Dict, Any, Tuple
from urllib.parse import urlsplit

from magic_llm.engine.base_chat import BaseChat
from magic_llm.engine.openai_adapters import (ProviderOpenAI,
                                              ProviderGroq,
                                              ProviderSambaNova,
                                              ProviderOpenRouter,
                                              ProviderMistral,
                                              ProviderFireworks,
                                              ProviderDeepseek,
                                              ProviderDeepInfra,
                                              ProviderTogether,
                                              OpenAiBaseProvider)
from magic_llm.engine.openai_adapters.responses import OpenAIResponsesAdapter
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel
from magic_llm.exception.ChatException import ProviderProtocolError
from magic_llm.util.http import AsyncHttpClient, HttpClient
from magic_llm.util.private_http import PrivateAsyncHttpClient, validate_private_provider_url
from magic_llm.util.private_limits import PrivateRequestLimits
from magic_llm.util.private_output import PrivateOutputGuard

logger = logging.getLogger(__name__)


class EngineOpenAI(BaseChat):
    engine = 'openai'
    external_embedding_control_version = 1

    # Match the parsed hostname; provider names in paths or host suffixes are
    # not permission to use another provider's adapter.
    PROVIDER_MAPPING = {
        ('api.groq.com', ''): ProviderGroq,
        ('api.sambanova.ai', ''): ProviderSambaNova,
        ('openrouter.ai', ''): ProviderOpenRouter,
        ('api.mistral.ai', ''): ProviderMistral,
        ('api.fireworks.ai', ''): ProviderFireworks,
        ('api.deepseek.com', ''): ProviderDeepseek,
        ('api.deepinfra.com', '/v1'): ProviderDeepInfra,
        ('api.together.xyz', ''): ProviderTogether,
    }

    def __init__(self,
                 api_key: str,
                 openai_adapter: Optional[Callable] = None,
                 base_url: Optional[str] = None,
                 endpoint: str = "chat_completions",
                 private_transport: bool = False,
                 private_limits: PrivateRequestLimits | None = None,
                 **kwargs) -> None:
        """
        Initialize the OpenAI engine with the appropriate provider.

        Args:
            api_key: The API key for authentication
            openai_adapter: Optional custom adapter class
            base_url: Optional base URL for the API
            **kwargs: Additional arguments for the provider
        """
        if endpoint not in {"chat_completions", "responses"}:
            raise ValueError("endpoint must be chat_completions or responses")
        self.endpoint = endpoint
        if type(private_transport) is not bool:
            raise ValueError('private_transport must be a boolean')
        self._private_transport = private_transport
        self._private_limits = private_limits
        if private_limits is not None:
            if not private_transport or not isinstance(private_limits, PrivateRequestLimits):
                raise ValueError('private_limits requires private transport and a PrivateRequestLimits profile')
            if kwargs.get('model') != private_limits.model or kwargs.get('fallback') is not None:
                raise ValueError('A limited request requires the approved model and cannot change providers')
        if private_transport:
            validate_private_provider_url(base_url or 'https://api.openai.com/v1')
            openai_adapter = ProviderOpenAI
        super().__init__(**kwargs)

        if private_limits is not None:
            # Keep generation defaults on the engine only. The policy resolves
            # them per attempt; the adapter must not reintroduce removed fields.
            kwargs = {'model': self.model}

        if openai_adapter is None and base_url is None:
            # Default to OpenAI if no adapter or base_url is provided
            self.base = ProviderOpenAI(api_key=api_key, **kwargs)
            logger.debug("Using default OpenAI provider")
        elif base_url:
            # An explicit adapter also applies when a custom base URL is set.
            provider_class = (openai_adapter
                if isinstance(openai_adapter, type) and issubclass(openai_adapter, OpenAiBaseProvider)
                else self._get_provider_for_url(base_url))
            if provider_class == ProviderOpenAI:
                self.base = ProviderOpenAI(api_key=api_key, base_url=base_url,
                                           stream_usage=not private_transport, **kwargs)
            else:
                self.base = provider_class(api_key=api_key, **kwargs)
                # Specialized adapters may hardcode a default URL. An explicit
                # base URL owns routing, including the provider's API prefix.
                self.base.base_url = base_url

            logger.debug(f"Selected provider {provider_class.__name__} for URL: {base_url}")
        elif isinstance(openai_adapter, type) and issubclass(openai_adapter, OpenAiBaseProvider):
            # Use the provided adapter class
            self.base = openai_adapter(api_key=api_key, **kwargs)
            logger.debug(f"Using custom adapter: {openai_adapter.__name__}")
        else:
            # Fallback to OpenAI
            self.base = ProviderOpenAI(api_key=api_key, **kwargs)
            logger.warning(f"Unrecognized adapter type: {type(openai_adapter)}. Using default OpenAI provider.")

        self._chat_provider = (OpenAIResponsesAdapter(self.base, include_encrypted_reasoning=not private_transport)
                               if endpoint == "responses" else self.base)

    @property
    def chat_url(self):
        path = "/responses" if self.endpoint == "responses" else "/chat/completions"
        return self.base.base_url.rstrip("/") + path

    def _prepare_provider_attempt(self, chat, options):
        if self._private_limits is None:
            return chat, options
        chat, effective, timeout = self._private_limits.prepare(chat, options)
        return chat, {**effective, 'timeout': timeout}

    def _output_guard(self):
        return closing(PrivateOutputGuard(self.base.api_key)) if self._private_transport else nullcontext()

    def _discovery_adapter(self):
        if self._private_transport:
            # Listing uses the ordinary HTTP client, outside the guarded transport
            # that protects a caller-supplied endpoint and key.
            raise NotImplementedError(
                "Model listing is not supported with private_transport; pass the model ID explicitly."
            )
        return super()._discovery_adapter()

    def _get_provider_for_url(self, url: str) -> Type[OpenAiBaseProvider]:
        """
        Determine the appropriate provider class based on the URL.

        Args:
            url: The base URL for the API

        Returns:
            The provider class to use
        """
        parsed = urlsplit(url)
        path = parsed.path.rstrip('/')

        for (hostname, prefix), provider in self.PROVIDER_MAPPING.items():
            if parsed.hostname == hostname and (not prefix or path == prefix or path.startswith(prefix + '/')):
                return provider

        # Default to OpenAI if no match is found
        return ProviderOpenAI

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMATION METHODS (Delegate to provider)
    # ═══════════════════════════════════════════════════════════════════

    def transform_request(
        self,
        chat: ModelChat,
        **kwargs
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to OpenAI-compatible request format.
        Delegates to the provider's transform_request method.

        Note: Image support depends on the underlying provider.
        Most OpenAI-compatible providers support images through the
        image_url content type.
        """
        if self._private_limits is not None:
            chat, kwargs = self._prepare_provider_attempt(chat, {**self.kwargs, **kwargs})
        return self._chat_provider.transform_request(chat, **kwargs)

    def transform_response(self, raw: Dict[str, Any]) -> ModelChatResponse:
        """
        Transform OpenAI API response to ModelChatResponse.
        Delegates to the provider's transform_response method.
        """
        return self._chat_provider.transform_response(raw)

    def transform_stream_chunk(
        self,
        raw: Any,
        context: Optional[Dict] = None
    ) -> Optional[ChatCompletionModel]:
        """
        Transform streaming chunk to ChatCompletionModel.
        Delegates to the provider's transform_stream_chunk method.
        """
        return self._chat_provider.transform_stream_chunk(raw, context if context is not None else {})

    def prepare_response(self, r):
        """Backward compatible alias for transform_response."""
        return self.transform_response(r)

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        json_data, headers = self._chat_provider.transform_request(chat, **kwargs)
        async with (PrivateAsyncHttpClient() if self._private_transport else AsyncHttpClient()) as client:
            response = await client.post_json(url=self.chat_url,
                                              data=json_data,
                                              headers=headers,
                                              timeout=kwargs.get('timeout', 30))
            with self._output_guard() as guard:
                if guard is not None:
                    guard.assert_safe(response)
                return self.transform_response(response)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        if self._private_transport:
            raise ValueError('Protected custom providers require asynchronous execution')
        # Make the request and read the response.
        data, headers = self._chat_provider.transform_request(chat, **kwargs)
        with HttpClient() as client:
            response = client.post_json(url=self.chat_url,
                                        data=data,
                                        headers=headers,
                                        timeout=kwargs.get('timeout', 30))
            return self.transform_response(response)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        if self._private_transport:
            raise ValueError('Protected custom providers require asynchronous execution')
        data, headers = self._chat_provider.transform_request(chat, stream=True, **kwargs)
        with HttpClient() as client:
            id_generation = ''
            context = {}
            for chunk in client.stream_request("POST",
                                               self.chat_url,
                                               data=data,
                                               headers=headers,
                                               timeout=kwargs.get('timeout', 30)):
                if c := self.transform_stream_chunk(chunk.strip(), context):
                    if c.id:
                        id_generation = c.id
                    context.update(id_generation=id_generation, last_chunk=c)
                    yield c

            if self.endpoint == "responses" and not context.get("terminal"):
                raise ProviderProtocolError("Responses stream ended before a terminal event")


    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        json_data, headers = self._chat_provider.transform_request(chat, stream=True, **kwargs)
        with self._output_guard() as guard:
            async with (PrivateAsyncHttpClient() if self._private_transport else AsyncHttpClient()) as client:
                id_generation = ''
                context = {}
                async for chunk in client.post_stream(self.chat_url, data=json_data, headers=headers,
                                                      timeout=kwargs.get('timeout', 30)):
                    frame = chunk.decode('utf-8').strip()
                    if guard is not None:
                        # Also inspect metadata dropped by the protocol adapter.
                        guard.assert_safe(frame)
                    if c := self.transform_stream_chunk(frame, context):
                        if guard is not None:
                            c = guard.guard_chunk(c)
                        if c.id:
                            id_generation = c.id
                        context.update(id_generation=id_generation, last_chunk=c)
                        yield c

                if self.endpoint == "responses" and not context.get("terminal"):
                    raise ProviderProtocolError("Responses stream ended before a terminal event")


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
            return self.base.transform_embedding_response(response)

    async def async_embedding(self, text: list[str] | str, *,
                              external_dispatch=None,
                              external_kind: str = 'memory.embedding.query',
                              **kwargs):
        if external_dispatch is not None:
            # Admission sees the exact resolved physical request. Control ports
            # are trusted host objects and never belong to the provider payload.
            timeout = kwargs.pop('timeout', 30)
            if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                    or not 0 < timeout <= 300):
                raise ValueError('Controlled embeddings require a finite timeout <= 300 seconds')
            data = {'input': text, 'model': self.model, **kwargs}
            body = json.dumps(data, allow_nan=False)
            if len(body.encode('utf-8')) > 4 * 1024 * 1024:
                raise ValueError('Controlled embedding request exceeds its byte ceiling')
            # Detach all mutable request inputs before calling any host callback.
            data = json.loads(body)
            headers = dict(self.base.headers)
            url = self.base.base_url + '/embeddings'
            max_response_bytes = 4 * 1024 * 1024
            request = {'method': 'POST', 'url': url, 'headers': dict(headers),
                       'json': data, 'timeout_seconds': timeout,
                       'allow_redirects': False, 'auto_decompress': False,
                       'max_request_bytes': 4 * 1024 * 1024, 'max_response_bytes': max_response_bytes}

            async def dispatch():
                async with AsyncHttpClient() as client:
                    response = await client.post_json(
                        url=url, data=body, headers=headers, timeout=timeout,
                        allow_redirects=False, max_response_bytes=max_response_bytes)
                    return self.base.transform_embedding_response(response)

            return await external_dispatch.call(external_kind, request, dispatch)

        async with AsyncHttpClient() as client:
            data = {
                "input": text,
                "model": self.model,
                **kwargs
            }
            response = await client.post_json(url=self.base.base_url + '/embeddings',
                                              data=json.dumps(data),
                                              headers=self.base.headers,
                                              timeout=kwargs.get('timeout', 30))
            return self.base.transform_embedding_response(response)

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        """
        Generate audio speech asynchronously.

        Args:
            data: The audio speech request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio speech response
        """
        return await self._retry_async_media(
            lambda: self.base.async_audio_speech(data, **kwargs),
            method='async_audio_speech',
        )

    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        """
        Generate audio speech synchronously.

        Args:
            data: The audio speech request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio speech response
        """
        return self._retry_sync_media(
            lambda: self.base.audio_speech(data, **kwargs),
            method='audio_speech',
        )

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        """
        Generate audio transcriptions asynchronously.

        Args:
            data: The audio transcriptions request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio transcriptions response
        """
        return await self._retry_async_media(
            lambda: self.base.async_audio_transcriptions(data, **kwargs),
            method='async_audio_transcriptions',
        )

    def sync_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        """
        Generate audio transcriptions synchronously.

        Args:
            data: The audio transcriptions request data
            **kwargs: Additional arguments for the request

        Returns:
            The audio transcriptions response
        """
        return self._retry_sync_media(
            lambda: self.base.sync_audio_transcriptions(data, **kwargs),
            method='sync_audio_transcriptions',
        )
