"""Tests for Fireworks ASR hostname updates."""

import pytest

from magic_llm.exception.ChatException import ChatException
from magic_llm.engine.openai_adapters.openai_fireworks import ProviderFireworks
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest


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

    async def post_multipart(self, **kwargs):
        self.calls.append(kwargs)
        return {'text': 'hello'}


class TestFireworksHostnames:
    """Test that Fireworks uses current *.api.fireworks.ai hostnames for ASR."""

    def test_whisper_v3_uses_current_hostname_in_source(self):
        """whisper-v3 ASR URL uses audio-prod.api.fireworks.ai."""
        assert ProviderFireworks.ASR_HOSTS["whisper-v3"] == "https://audio-prod.api.fireworks.ai/v1"

    def test_whisper_v3_turbo_uses_current_hostname_in_source(self):
        """whisper-v3-turbo ASR URL uses audio-turbo.api.fireworks.ai."""
        assert ProviderFireworks.ASR_HOSTS["whisper-v3-turbo"] == "https://audio-turbo.api.fireworks.ai/v1"

    def test_base_url_is_general_inference_endpoint(self):
        """Provider base_url is the general inference endpoint (not ASR-specific)."""
        provider = ProviderFireworks(api_key="test")
        assert provider.base_url == "https://api.fireworks.ai/inference/v1"

    @pytest.mark.asyncio
    async def test_whisper_v3_request_hits_current_hostname(self, monkeypatch):
        import magic_llm.engine.openai_adapters.openai_fireworks as mod

        CapturingAsyncClient.instances.clear()
        monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)
        provider = ProviderFireworks(api_key='test', model='whisper-v3')
        request = AudioTranscriptionsRequest(
            file=WAV_BYTES,
            model='whisper-v3',
            filename='sample.wav',
            content_type='audio/wav',
        )

        assert await provider.async_audio_transcriptions(request) == {'text': 'hello'}
        call = CapturingAsyncClient.instances[0].calls[0]
        assert call['url'] == 'https://audio-prod.api.fireworks.ai/v1/audio/transcriptions'
        assert call['fields']['model'] == 'whisper-v3'
        assert call['filename'] == 'sample.wav'
        assert call['content_type'] == 'audio/wav'

    @pytest.mark.asyncio
    async def test_unknown_asr_model_rejects_before_http(self, monkeypatch):
        import magic_llm.engine.openai_adapters.openai_fireworks as mod

        CapturingAsyncClient.instances.clear()
        monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)
        provider = ProviderFireworks(api_key='test', model='not-supported')
        request = AudioTranscriptionsRequest(
            file=WAV_BYTES,
            filename='sample.wav',
            content_type='audio/wav',
        )

        with pytest.raises(ChatException) as exc_info:
            await provider.async_audio_transcriptions(request)
        assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'
        assert CapturingAsyncClient.instances == []
