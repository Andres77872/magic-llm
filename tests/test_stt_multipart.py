import json
import asyncio
from unittest.mock import MagicMock

import aiohttp
import pytest
from pydantic import ValidationError

import magic_llm.model.ModelAudio as model_audio
from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.exception.ChatException import ChatException
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest
from magic_llm.util.http import AsyncHttpClient, HttpClient


WAV_BYTES = b'RIFF\x10\x00\x00\x00WAVEfmt '


class FakeResponse:
    status_code = 200
    content = b'{"text": "hello"}'


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeResponse()


class AsyncResponse:
    status = 200

    async def read(self):
        return b'{"text": "hello"}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


class AsyncFakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return AsyncResponse()


def test_request_preserves_explicit_filename_and_content_type():
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-1',
        filename='sample.wav',
        content_type='audio/wav',
    )
    metadata = request.resolve_upload_metadata()
    assert metadata.filename == 'sample.wav'
    assert metadata.content_type == 'audio/wav'


def test_request_rejects_filename_extension_conflicting_with_content_type():
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-1',
        filename='sample.mp3',
        content_type='audio/wav',
    )
    with pytest.raises(ValueError, match='filename extension'):
        request.resolve_upload_metadata()


def test_request_rejects_content_type_conflicting_with_inferred_bytes():
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-1',
        filename='sample.wav',
        content_type='audio/mpeg',
    )
    with pytest.raises(ValueError, match='Audio bytes appear to be audio/wav'):
        request.resolve_upload_metadata()


def test_request_infers_wav_metadata_from_bytes():
    request = AudioTranscriptionsRequest(file=WAV_BYTES, model='whisper-1')
    metadata = request.resolve_upload_metadata()
    assert metadata.filename == 'audio.wav'
    assert metadata.content_type == 'audio/wav'


def test_unknown_audio_metadata_fails_conservatively():
    request = AudioTranscriptionsRequest(file=b'not-known-audio', model='whisper-1')
    with pytest.raises(ValueError, match='Unable to infer'):
        request.resolve_upload_metadata()


def test_empty_and_oversized_audio_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        AudioTranscriptionsRequest(file=b'', model='whisper-1')

    monkeypatch.setattr(model_audio, 'DEFAULT_MAX_AUDIO_UPLOAD_BYTES', 4)
    with pytest.raises(ValidationError):
        AudioTranscriptionsRequest(file=b'12345', model='whisper-1')


def test_sync_post_multipart_uses_requests_files_and_data():
    session = FakeSession()
    client = HttpClient()
    client.session = session

    response = client.post_multipart(
        url='https://example.test/audio/transcriptions',
        fields={'model': 'whisper-1', 'response_format': 'json', 'temperature': '0'},
        file_field='file',
        file_bytes=WAV_BYTES,
        filename='sample.wav',
        content_type='audio/wav',
        headers={'Authorization': 'Bearer test', 'Content-Type': 'application/json'},
    )

    assert response == {'text': 'hello'}
    _, _, kwargs = session.calls[0]
    assert kwargs['data']['model'] == 'whisper-1'
    assert kwargs['data']['response_format'] == 'json'
    assert kwargs['files']['file'] == ('sample.wav', WAV_BYTES, 'audio/wav')
    assert 'Content-Type' not in kwargs['headers']


@pytest.mark.parametrize('header_name', ['Content-Type', 'content-type', 'CONTENT-TYPE'])
def test_sync_post_multipart_strips_content_type_header_case_insensitively(header_name):
    session = FakeSession()
    client = HttpClient()
    client.session = session

    client.post_multipart(
        url='https://example.test/audio/transcriptions',
        fields={'model': 'whisper-1'},
        file_field='file',
        file_bytes=WAV_BYTES,
        filename='sample.wav',
        content_type='audio/wav',
        headers={'Authorization': 'Bearer test', header_name: 'multipart/form-data'},
    )

    _, _, kwargs = session.calls[0]
    assert header_name not in kwargs['headers']
    assert not any(key.lower() == 'content-type' for key in kwargs['headers'])


def test_async_post_multipart_uses_aiohttp_formdata():
    session = AsyncFakeSession()
    client = AsyncHttpClient()
    client.session = session

    async def _call():
        return await client.post_multipart(
            url='https://example.test/audio/transcriptions',
            fields={'model': 'whisper-1', 'language': 'en'},
            file_field='file',
            file_bytes=WAV_BYTES,
            filename='sample.wav',
            content_type='audio/wav',
            headers={'Authorization': 'Bearer test', 'Content-Type': 'application/json'},
        )

    response = asyncio.run(_call())

    assert response == {'text': 'hello'}
    _, _, kwargs = session.calls[0]
    assert isinstance(kwargs['data'], aiohttp.FormData)
    assert 'Content-Type' not in kwargs['headers']


@pytest.mark.parametrize('response_format', ['text', 'srt', 'vtt'])
def test_text_like_response_formats_bypass_json_decoding(response_format):
    session = FakeSession()
    session.request = MagicMock(return_value=type('Resp', (), {'status_code': 200, 'content': b'plain transcript'})())
    client = HttpClient()
    client.session = session

    response = client.post_multipart(
        url='https://example.test/audio/transcriptions',
        fields={'model': 'whisper-1', 'response_format': response_format},
        file_field='file',
        file_bytes=WAV_BYTES,
        filename='sample.wav',
        content_type='audio/wav',
        response_format=response_format,
    )
    assert response == 'plain transcript'


def test_verbose_json_response_normalizes_transcript_text():
    provider = OpenAiBaseProvider(base_url='https://example.test/v1', api_key='test')
    normalized = provider._normalize_transcription_response({'transcript': 'hello', 'duration': 1}, 'verbose_json')
    assert normalized['text'] == 'hello'
    assert normalized['duration'] == 1


def test_transcription_fields_include_optional_metadata_and_model_override():
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        language='en',
        prompt='domain words',
        response_format='verbose_json',
        temperature=0.2,
    )
    assert request.transcription_fields(model='whisper-1') == {
        'model': 'whisper-1',
        'language': 'en',
        'prompt': 'domain words',
        'response_format': 'verbose_json',
        'temperature': '0.2',
    }


def test_unsupported_openai_compatible_stt_fails_before_http(monkeypatch):
    provider = OpenAiBaseProvider(base_url='https://example.test/v1', api_key='test', model='chat-only')
    request = AudioTranscriptionsRequest(
        file=WAV_BYTES,
        model='whisper-1',
        filename='sample.wav',
        content_type='audio/wav',
    )
    with pytest.raises(ChatException) as exc_info:
        provider.sync_audio_transcriptions(request)
    assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'


def test_json_response_normalization_preserves_provider_fields():
    provider = OpenAiBaseProvider(base_url='https://example.test/v1', api_key='test')
    normalized = provider._normalize_transcription_response({'DisplayText': 'hello', 'raw': True}, 'json')
    assert normalized['text'] == 'hello'
    assert normalized['raw'] is True
