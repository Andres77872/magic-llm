"""Offline Google TTS payload, headers, and WAV conversion proof."""

import base64
import json

import pytest

from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.model.ModelAudio import AudioSpeechRequest


PCM_BYTES = b'\x01\x02\x03\x04' * 20
GOOGLE_TTS_RESPONSE = {
    'candidates': [{
        'content': {
            'parts': [{
                'inlineData': {'data': base64.b64encode(PCM_BYTES).decode()}
            }]
        }
    }]
}


class CapturingSyncClient:
    instances = []

    def __init__(self):
        self.calls = []
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def post_json(self, *, url, data=None, json=None, headers=None, timeout=30, **kwargs):
        self.calls.append({'url': url, 'data': data, 'json': json, 'headers': headers, 'timeout': timeout})
        return GOOGLE_TTS_RESPONSE


class CapturingAsyncClient(CapturingSyncClient):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post_json(self, *, url, data=None, json=None, headers=None, timeout=30, **kwargs):
        self.calls.append({'url': url, 'data': data, 'json': json, 'headers': headers, 'timeout': timeout})
        return GOOGLE_TTS_RESPONSE


def _speech_request():
    return AudioSpeechRequest(
        input='Say hello',
        model='gemini-2.5-flash-preview-tts',
        voice='Kore',
        response_format='wav',
    )


def test_prepare_tts_data_includes_text_voice_and_model():
    engine = EngineGoogle(api_key='dummy-key', model='gemini-2.5-flash')
    payload = engine._prepare_tts_data(_speech_request())

    assert payload['contents'][0]['parts'][0]['text'] == 'Say hello'
    assert payload['generationConfig']['responseModalities'] == ['AUDIO']
    assert payload['generationConfig']['speechConfig']['voiceConfig']['prebuiltVoiceConfig']['voiceName'] == 'Kore'
    assert payload['model'] == 'gemini-2.5-flash-preview-tts'


def test_pcm_to_wav_bytes_returns_wav_container():
    engine = EngineGoogle(api_key='dummy-key', model='gemini-2.5-flash')
    wav = engine._pcm_to_wav_bytes(PCM_BYTES)

    assert wav.startswith(b'RIFF')
    assert b'WAVE' in wav[:16]
    assert len(wav) > len(PCM_BYTES)


def test_google_sync_tts_payload_headers_and_wav_response(monkeypatch):
    import magic_llm.engine.engine_google as mod

    CapturingSyncClient.instances.clear()
    monkeypatch.setattr(mod, 'HttpClient', CapturingSyncClient)
    engine = EngineGoogle(api_key='dummy-key', model='gemini-2.5-flash')

    wav = engine.audio_speech(_speech_request(), timeout=17)

    assert wav.startswith(b'RIFF')
    call = CapturingSyncClient.instances[0].calls[0]
    assert call['url'].endswith('/gemini-2.5-flash-preview-tts:generateContent')
    payload = json.loads(call['data'].decode('utf-8'))
    assert payload['contents'][0]['parts'][0]['text'] == 'Say hello'
    assert payload['generationConfig']['speechConfig']['voiceConfig']['prebuiltVoiceConfig']['voiceName'] == 'Kore'
    assert call['headers']['Content-Type'] == 'application/json'
    assert call['headers']['x-goog-api-key'] == 'dummy-key'
    assert call['timeout'] == 17


@pytest.mark.asyncio
async def test_google_async_tts_payload_headers_and_wav_response(monkeypatch):
    import magic_llm.engine.engine_google as mod

    CapturingAsyncClient.instances.clear()
    monkeypatch.setattr(mod, 'AsyncHttpClient', CapturingAsyncClient)
    engine = EngineGoogle(api_key='dummy-key', model='gemini-2.5-flash')

    wav = await engine.async_audio_speech(_speech_request(), timeout=19)

    assert wav.startswith(b'RIFF')
    call = CapturingAsyncClient.instances[0].calls[0]
    assert call['url'].endswith('/gemini-2.5-flash-preview-tts:generateContent')
    payload = json.loads(call['data'].decode('utf-8'))
    assert payload['contents'][0]['parts'][0]['text'] == 'Say hello'
    assert call['headers']['x-goog-api-key'] == 'dummy-key'
    assert call['timeout'] == 19
