"""Offline provider media request-shape proof for OpenAI-compatible adapters."""

import base64

import pytest

from magic_llm.engine.openai_adapters.openai_base import ProviderOpenAI
from magic_llm.engine.openai_adapters.openai_deepinfra import ProviderDeepInfra
from magic_llm.engine.openai_adapters.openai_fireworks import ProviderFireworks
from magic_llm.engine.openai_adapters.openai_together import ProviderTogether
from magic_llm.exception.ChatException import ChatException
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest


WAV_BYTES = b'RIFF\x10\x00\x00\x00WAVEfmt '


class CapturingAsyncClient:
    instances = []

    def __init__(self):
        self.calls = []
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post_raw_binary(self, *, url, json=None, data=None, headers=None, timeout=30, **kwargs):
        self.calls.append({
            'method': 'post_raw_binary',
            'url': url,
            'json': json,
            'data': data,
            'headers': headers,
            'timeout': timeout,
            'kwargs': kwargs,
        })
        return b'synthetic-audio'

    async def post_json(self, *, url, json=None, data=None, headers=None, timeout=30, **kwargs):
        self.calls.append({
            'method': 'post_json',
            'url': url,
            'json': json,
            'data': data,
            'headers': headers,
            'timeout': timeout,
            'kwargs': kwargs,
        })
        return {'audio': 'data:audio/wav;base64,' + base64.b64encode(b'RIFFdeepinfra').decode()}

    async def post_multipart(
        self,
        *,
        url,
        fields,
        file_field,
        file_bytes,
        filename,
        content_type,
        headers=None,
        response_format='json',
        timeout=30,
        **kwargs,
    ):
        self.calls.append({
            'method': 'post_multipart',
            'url': url,
            'fields': fields,
            'file_field': file_field,
            'file_bytes': file_bytes,
            'filename': filename,
            'content_type': content_type,
            'headers': headers,
            'response_format': response_format,
            'timeout': timeout,
            'kwargs': kwargs,
        })
        return {'text': 'hello'}


class CapturingSyncClient:
    instances = []

    def __init__(self):
        self.calls = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def post_raw_binary(self, *, url, json=None, data=None, headers=None, timeout=30, **kwargs):
        self.calls.append({
            'method': 'post_raw_binary',
            'url': url,
            'json': json,
            'data': data,
            'headers': headers,
            'timeout': timeout,
            'kwargs': kwargs,
        })
        return b'synthetic-audio'


def _speech_request(model='tts-1', voice='alloy'):
    return AudioSpeechRequest(input='Hello from offline proof', model=model, voice=voice, response_format='mp3')


@pytest.mark.asyncio
async def test_openai_async_tts_request_shape(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_base as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)

    provider = ProviderOpenAI(api_key='dummy-key', model='gpt-4o-mini-tts')
    result = await provider.async_audio_speech(_speech_request())

    assert result == b'synthetic-audio'
    call = CapturingAsyncClient.instances[0].calls[0]
    assert call['url'] == 'https://api.openai.com/v1/audio/speech'
    assert call['json']['input'] == 'Hello from offline proof'
    assert call['json']['model'] == 'tts-1'
    assert call['json']['voice'] == 'alloy'
    assert call['headers']['Authorization'].startswith('Bearer ')


def test_together_sync_tts_request_shape(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_together as mod

    CapturingSyncClient.instances.clear()
    monkeypatch.setattr(mod, 'HttpClient', CapturingSyncClient)

    provider = ProviderTogether(api_key='dummy-key', model='cartesia/sonic-2')
    result = provider.audio_speech(_speech_request(model='cartesia/sonic-2', voice='narrator'))

    assert result == b'synthetic-audio'
    call = CapturingSyncClient.instances[0].calls[0]
    assert call['url'] == 'https://api.together.xyz/v1/audio/generations'
    assert call['json']['model'] == 'cartesia/sonic-2'
    assert call['json']['input'] == 'Hello from offline proof'
    assert call['json']['voice'] == 'narrator'
    assert call['json']['response_encoding'] == 'pcm_f32le'
    assert call['headers']['Authorization'].startswith('Bearer ')


@pytest.mark.asyncio
async def test_together_async_tts_request_shape(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_together as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)

    provider = ProviderTogether(api_key='dummy-key', model='cartesia/sonic-2')
    result = await provider.async_audio_speech(_speech_request(model='cartesia/sonic-2', voice='narrator'))

    assert result == b'synthetic-audio'
    call = CapturingAsyncClient.instances[0].calls[0]
    assert call['url'] == 'https://api.together.xyz/v1/audio/generations'
    assert call['json']['sample_rate'] == 44100
    assert call['json']['stream'] is False


@pytest.mark.asyncio
async def test_deepinfra_async_tts_request_shape_and_response_decode(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_deepinfra as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)

    provider = ProviderDeepInfra(api_key='dummy-key', model='ignored')
    result = await provider.async_audio_speech(_speech_request(model='deepinfra-tts-model', voice='voice-1'))

    assert result == b'RIFFdeepinfra'
    call = CapturingAsyncClient.instances[0].calls[0]
    assert call['url'] == 'https://api.deepinfra.com/v1/inference/deepinfra-tts-model'
    assert call['json'] == {'text': 'Hello from offline proof', 'voice_id': 'voice-1'}
    assert call['headers']['Authorization'].startswith('Bearer ')


@pytest.mark.asyncio
async def test_fireworks_async_stt_multipart_request_shape(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_fireworks as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)

    provider = ProviderFireworks(api_key='dummy-key', model='whisper-v3')
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-v3',
        filename='sample.wav',
        content_type='audio/wav',
        response_format='verbose_json',
        language='en',
    )
    result = await provider.async_audio_transcriptions(request, timeout=12)

    assert result == {'text': 'hello'}
    call = CapturingAsyncClient.instances[0].calls[0]
    assert call['url'] == 'https://audio-prod.api.fireworks.ai/v1/audio/transcriptions'
    assert call['fields']['model'] == 'whisper-v3'
    assert call['fields']['response_format'] == 'verbose_json'
    assert call['fields']['language'] == 'en'
    assert call['file_field'] == 'file'
    assert call['filename'] == 'sample.wav'
    assert call['content_type'] == 'audio/wav'
    assert call['response_format'] == 'json'
    assert call['timeout'] == 12
    assert call['headers']['Authorization'].startswith('Bearer ')


@pytest.mark.asyncio
async def test_fireworks_unknown_model_rejects_before_http(monkeypatch):
    import magic_llm.engine.openai_adapters.openai_fireworks as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)

    provider = ProviderFireworks(api_key='dummy-key', model='not-an-asr-model')
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        filename='sample.wav',
        content_type='audio/wav',
    )
    with pytest.raises(ChatException) as exc_info:
        await provider.async_audio_transcriptions(request)

    assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'
    assert 'Supported ASR models' in exc_info.value.message
    assert CapturingAsyncClient.instances == []
