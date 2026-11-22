"""Offline provider media support-flag and fail-fast matrix."""

import json

import pytest

from magic_llm.engine.openai_adapters.openai_base import ProviderOpenAI
from magic_llm.engine.openai_adapters.openai_deepinfra import ProviderDeepInfra
from magic_llm.engine.openai_adapters.openai_fireworks import ProviderFireworks
from magic_llm.engine.openai_adapters.openai_groq import ProviderGroq
from magic_llm.engine.openai_adapters.openai_openrouter import ProviderOpenRouter
from magic_llm.engine.openai_adapters.openai_together import ProviderTogether
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest


WAV_BYTES = b'RIFF\x10\x00\x00\x00WAVEfmt '
PNG_1x1_BASE64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='
)


def _transcription():
    return AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-1',
        filename='sample.wav',
        content_type='audio/wav',
    )


@pytest.mark.parametrize(
    ('label', 'provider', 'expected'),
    [
        ('openai', ProviderOpenAI(api_key='dummy', model='gpt-4o'), {
            'supports_tts_sync': False,
            'supports_tts_async': True,
            'supports_stt_sync': True,
            'supports_stt_async': True,
            'supports_vision': True,
        }),
        ('together', ProviderTogether(api_key='dummy', model='cartesia/sonic-2'), {
            'supports_tts_sync': True,
            'supports_tts_async': True,
            'supports_stt_sync': False,
            'supports_stt_async': False,
            'supports_vision': False,
        }),
        ('deepinfra', ProviderDeepInfra(api_key='dummy', model='tts-model'), {
            'supports_tts_sync': False,
            'supports_tts_async': True,
            'supports_stt_sync': False,
            'supports_stt_async': False,
            'supports_vision': False,
        }),
        ('fireworks', ProviderFireworks(api_key='dummy', model='whisper-v3'), {
            'supports_tts_sync': False,
            'supports_tts_async': False,
            'supports_stt_sync': False,
            'supports_stt_async': True,
            'supports_vision': False,
        }),
        ('groq_non_supporting', ProviderGroq(api_key='dummy', model='llama3'), {
            'supports_tts_sync': False,
            'supports_tts_async': False,
            'supports_stt_sync': False,
            'supports_stt_async': False,
            'supports_vision': False,
        }),
        ('openrouter', ProviderOpenRouter(api_key='dummy', model='openai/gpt-4o'), {
            'supports_tts_sync': False,
            'supports_tts_async': False,
            'supports_stt_sync': False,
            'supports_stt_async': False,
            'supports_vision': True,
        }),
    ],
    ids=['openai', 'together', 'deepinfra', 'fireworks', 'groq_non_supporting', 'openrouter'],
)
def test_openai_compatible_media_support_flags(label, provider, expected):
    for attr, value in expected.items():
        assert getattr(provider, attr) is value, f'{label}: {attr}'


def test_openrouter_forwards_image_input():
    provider = ProviderOpenRouter(api_key='dummy', model='openai/gpt-4o')
    chat = ModelChat()
    chat.add_user_message('Describe', image=f'data:image/png;base64,{PNG_1x1_BASE64}')

    body, _ = provider.transform_request(chat)

    content = json.loads(body)['messages'][-1]['content']
    assert any(part['type'] == 'image_url' for part in content)


def test_custom_openai_compatible_url_downgrades_unverified_media_and_vision():
    provider = ProviderOpenAI(api_key='dummy', base_url='https://proxy.example.test/v1', model='gpt-4o')
    assert provider.supports_vision is False
    assert provider.supports_tts_async is False
    assert provider.supports_stt_sync is False
    assert provider.supports_stt_async is False

    with pytest.raises(ChatException) as exc_info:
        provider.sync_audio_transcriptions(_transcription())

    assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'


def test_unsupported_openai_compatible_media_fails_before_http(monkeypatch):
    import magic_llm.engine.openai_adapters.base_provider as mod

    created = []

    class ExplodingClient:
        def __enter__(self):
            created.append('http')
            raise AssertionError('HTTP client must not be opened for unsupported media')

    monkeypatch.setattr(mod, 'HttpClient', ExplodingClient)
    provider = ProviderGroq(api_key='dummy', model='llama3')

    with pytest.raises(ChatException) as exc_info:
        provider.sync_audio_transcriptions(_transcription())

    assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'
    assert created == []
