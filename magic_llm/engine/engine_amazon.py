import base64
import json
from typing import Dict, Any, Tuple, Optional

from yarl import URL

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
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.http import AsyncHttpClient, HttpClient, HttpError
from magic_llm.util.sigv4 import (
    build_sigv4_headers,
    build_sigv4_prepared_request,
    build_bedrock_url,
    build_polly_url,
    resolve_credentials,
    resolve_credentials_async,
)
from magic_llm.util.eventstream import AWSEventStreamParser


class EngineAmazon(BaseChat):
    engine = 'amazon'

    def __init__(self,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
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

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMATION METHODS (Delegate to provider)
    # ═══════════════════════════════════════════════════════════════════

    def transform_request(
        self,
        chat: ModelChat,
        **kwargs
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Transform ModelChat to Amazon Bedrock request format.
        Delegates to the provider's transform_request method.

        Note: Returns (body_bytes, headers) for consistency with other engines,
        though Amazon uses body as string and separate API call patterns.

        Image support depends on the underlying provider:
        - Nova: Supports images via content array format
        - Titan: No image support
        - Anthropic (Bedrock): Legacy format, no image support
        - Meta: No image support
        """
        body = self.provider.transform_request(chat, **kwargs)
        headers = {
            'accept': 'application/json',
            'contentType': 'application/json'
        }
        return body.encode('utf-8'), headers

    def transform_response(self, raw: Dict[str, Any]) -> ModelChatResponse:
        """
        Transform Amazon Bedrock response to ModelChatResponse.
        Delegates to the provider's transform_response method.
        """
        return self.provider.transform_response(raw)

    def transform_stream_chunk(
        self,
        raw: Any,
        context: Optional[Dict] = None
    ) -> Optional[ChatCompletionModel]:
        """
        Transform streaming chunk to ChatCompletionModel.
        Delegates to the provider's transform_stream_chunk method.
        """
        return self.provider.transform_stream_chunk(raw)

    @staticmethod
    def _decode_bedrock_chunk(payload: dict) -> dict:
        """Decode Bedrock streaming chunk payload.

        Bedrock wraps provider events in a ``bytes`` field containing
        base64-encoded JSON. This method extracts and decodes it.

        If the payload has no ``bytes`` key, it is returned as-is
        (backward compatibility with non-Bedrock EventStream sources).
        """
        raw_bytes = payload.get('bytes')
        if raw_bytes is None:
            return payload
        # raw_bytes may already be a str (from JSON parsing) or bytes
        if isinstance(raw_bytes, bytes):
            raw_bytes = raw_bytes.decode('utf-8')
        decoded = base64.b64decode(raw_bytes)
        return json.loads(decoded.decode('utf-8'))

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Resolve credentials off the event loop (may block on IMDS/STS for ambient creds)
        access_key, secret_key, region, session_token = await resolve_credentials_async(
            self.provider.aws_access_key_id,
            self.provider.aws_secret_access_key,
            self.provider.region_name,
        )

        # Build request body and SigV4-signed prepared request
        body = self.provider.transform_request(chat, **kwargs)
        url = build_bedrock_url(region, self.model, stream=False)
        prepared = build_sigv4_prepared_request(
            method='POST',
            url=url,
            body=body,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region=region,
            service='bedrock',
            session_token=session_token,
            extra_headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
        )

        # Extract transport-ready headers from the prepared request
        aiohttp_headers = dict(prepared.headers)

        # Execute raw HTTP request — use URL(encoded=True) to prevent yarl requoting
        timeout = kwargs.get('timeout', 120)
        async with AsyncHttpClient() as client:
            raw_bytes = await client.post_json(
                url=URL(prepared.url, encoded=True),
                data=prepared.body,
                headers=aiohttp_headers,
                timeout=timeout,
            )
            return self.provider.transform_response(raw_bytes)

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Resolve credentials (explicit or ambient)
        access_key, secret_key, region, session_token = resolve_credentials(
            self.provider.aws_access_key_id,
            self.provider.aws_secret_access_key,
            self.provider.region_name,
        )

        # Build request body and SigV4-signed URL/headers
        body = self.provider.transform_request(chat, **kwargs)
        url = build_bedrock_url(region, self.model, stream=False)
        sigv4_headers = build_sigv4_headers(
            method='POST',
            url=url,
            body=body,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region=region,
            service='bedrock',
            session_token=session_token,
        )
        sigv4_headers['Content-Type'] = 'application/json'
        sigv4_headers['Accept'] = 'application/json'

        # Execute raw HTTP request
        timeout = kwargs.get('timeout', 120)
        with HttpClient() as client:
            raw_bytes = client.post_json(
                url=url,
                data=body,
                headers=sigv4_headers,
                timeout=timeout,
            )
            return self.provider.transform_response(raw_bytes)

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        # Resolve credentials (explicit or ambient)
        access_key, secret_key, region, session_token = resolve_credentials(
            self.provider.aws_access_key_id,
            self.provider.aws_secret_access_key,
            self.provider.region_name,
        )

        # Build request body and SigV4-signed URL/headers
        body = self.provider.transform_request(chat, **kwargs)
        url = build_bedrock_url(region, self.model, stream=True)
        sigv4_headers = build_sigv4_headers(
            method='POST',
            url=url,
            body=body,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region=region,
            service='bedrock',
            session_token=session_token,
        )
        sigv4_headers['Content-Type'] = 'application/json'
        sigv4_headers['Accept'] = 'application/json'

        # Execute streaming request and parse EventStream
        timeout = kwargs.get('timeout', 120)
        parser = AWSEventStreamParser()
        with HttpClient() as client:
            for raw_bytes in client.stream_request_bytes(
                "POST", url, data=body, headers=sigv4_headers, timeout=timeout
            ):
                for event in parser.feed(raw_bytes):
                    event_type = event.get(':event-type', '')
                    payload = event.get('payload')

                    if event_type == 'chunk' and isinstance(payload, dict):
                        # Bedrock wraps provider events in a 'bytes' field (base64-encoded JSON)
                        provider_event = self._decode_bedrock_chunk(payload)
                        chunk = self.provider.transform_stream_chunk(provider_event)
                        metrics = payload.get('amazon-bedrock-invocationMetrics', {})
                        prompt_tokens = metrics.get('inputTokenCount', 0)
                        completion_tokens = metrics.get('outputTokenCount', 0)
                        chunk.usage = UsageModel(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        yield chunk
                    elif event_type == 'exception':
                        # Log exception event and yield a final chunk with error info
                        error_info = payload if isinstance(payload, dict) else {}
                        raise HttpError(
                            f"Bedrock streaming exception: {error_info}",
                            response_content=str(error_info).encode(),
                        )

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        # Resolve credentials off the event loop (may block on IMDS/STS for ambient creds)
        access_key, secret_key, region, session_token = await resolve_credentials_async(
            self.provider.aws_access_key_id,
            self.provider.aws_secret_access_key,
            self.provider.region_name,
        )

        # Build request body and SigV4-signed prepared request
        body = self.provider.transform_request(chat, **kwargs)
        url = build_bedrock_url(region, self.model, stream=True)
        prepared = build_sigv4_prepared_request(
            method='POST',
            url=url,
            body=body,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region=region,
            service='bedrock',
            session_token=session_token,
            extra_headers={
                'Content-Type': 'application/json',
                'X-Amzn-Bedrock-Accept': 'application/json',
            },
        )

        # Extract transport-ready headers from the prepared request
        aiohttp_headers = dict(prepared.headers)

        # Execute streaming request and parse EventStream — use URL(encoded=True) to prevent yarl requoting
        timeout = kwargs.get('timeout', 120)
        parser = AWSEventStreamParser()
        async with AsyncHttpClient() as client:
            async for raw_bytes in client.post_stream(
                url=URL(prepared.url, encoded=True),
                data=prepared.body,
                headers=aiohttp_headers,
                timeout=timeout,
            ):
                for event in parser.feed(raw_bytes):
                    event_type = event.get(':event-type', '')
                    payload = event.get('payload')

                    if event_type == 'chunk' and isinstance(payload, dict):
                        # Bedrock wraps provider events in a 'bytes' field (base64-encoded JSON)
                        provider_event = self._decode_bedrock_chunk(payload)
                        chunk = self.provider.transform_stream_chunk(provider_event)
                        metrics = payload.get('amazon-bedrock-invocationMetrics', {})
                        prompt_tokens = metrics.get('inputTokenCount', 0)
                        completion_tokens = metrics.get('outputTokenCount', 0)
                        chunk.usage = UsageModel(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        yield chunk
                    elif event_type == 'exception':
                        error_info = payload if isinstance(payload, dict) else {}
                        raise HttpError(
                            f"Bedrock streaming exception: {error_info}",
                            response_content=str(error_info).encode(),
                        )

    def audio_speech(self, data: AudioSpeechRequest, **kwargs) -> bytes:
        """Generate speech via Amazon Polly using raw HTTP + SigV4 signing.

        Returns raw audio bytes (not an SDK AudioStream object).
        """
        # Resolve credentials (explicit or ambient)
        access_key, secret_key, region, session_token = resolve_credentials(
            self.provider.aws_access_key_id,
            self.provider.aws_secret_access_key,
            self.provider.region_name,
        )

        # Build Polly request body and SigV4-signed URL/headers
        polly_body = json.dumps({
            'VoiceId': data.voice,
            'OutputFormat': data.response_format,
            'Text': data.input,
            'Engine': data.model,
        })
        url = build_polly_url(region)
        sigv4_headers = build_sigv4_headers(
            method='POST',
            url=url,
            body=polly_body,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region=region,
            service='polly',
            session_token=session_token,
        )
        sigv4_headers['Content-Type'] = 'application/json'

        # Execute raw HTTP request — returns raw audio bytes
        timeout = kwargs.get('timeout', 30)
        with HttpClient() as client:
            return client.post_raw_binary(
                url=url,
                data=polly_body,
                headers=sigv4_headers,
                timeout=timeout,
            )
