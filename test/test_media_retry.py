import asyncio

import pytest

from magic_llm.engine.base_chat import BaseChat
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest
from magic_llm.util.http import HttpError


class RetryDummy(BaseChat):
    engine = 'retry-dummy'

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    def stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError

    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def recorded_retry_delays(monkeypatch):
    sync_delays = []
    async_delays = []

    async def record_async_delay(delay):
        async_delays.append(delay)

    monkeypatch.setattr("magic_llm.engine.base_chat.time.sleep", sync_delays.append)
    monkeypatch.setattr("magic_llm.engine.base_chat.asyncio.sleep", record_async_delay)
    return sync_delays, async_delays


def test_sync_media_retries_transient_http_failure(recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    attempts = []

    def operation():
        attempts.append('try')
        if len(attempts) == 1:
            raise HttpError('HTTP 500', status_code=500)
        return b'audio'

    assert dummy._retry_sync_media(operation, method='audio_speech') == b'audio'
    assert len(attempts) == 2
    assert recorded_retry_delays == ([1.0], [])


def test_async_media_retries_transient_http_failure(recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    attempts = []

    async def operation():
        attempts.append('try')
        if len(attempts) == 1:
            raise HttpError('HTTP 429', status_code=429)
        return {'text': 'ok'}

    assert asyncio.run(dummy._retry_async_media(operation, method='async_audio_transcriptions')) == {'text': 'ok'}
    assert len(attempts) == 2
    assert recorded_retry_delays == ([], [1.0])


@pytest.mark.parametrize('status_code', [None, 504])
def test_sync_media_retries_transport_error_and_gateway_timeout(status_code, recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    attempts = []

    def operation():
        attempts.append('try')
        if len(attempts) == 1:
            raise HttpError('transient media failure', status_code=status_code)
        return b'audio'

    assert dummy._retry_sync_media(operation, method='audio_speech') == b'audio'
    assert len(attempts) == 2
    assert recorded_retry_delays == ([1.0], [])


def test_async_media_retries_gateway_timeout(recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    attempts = []

    async def operation():
        attempts.append('try')
        if len(attempts) == 1:
            raise HttpError('HTTP 504', status_code=504)
        return {'text': 'ok'}

    assert asyncio.run(dummy._retry_async_media(operation, method='async_audio_transcriptions')) == {'text': 'ok'}
    assert len(attempts) == 2
    assert recorded_retry_delays == ([], [1.0])


def test_retry_exhaustion_surfaces_final_error(recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    attempts = []

    def operation():
        attempts.append('try')
        raise HttpError('HTTP 503', status_code=503)

    with pytest.raises(HttpError):
        dummy._retry_sync_media(operation, method='audio_speech')
    assert len(attempts) == 2
    assert recorded_retry_delays == ([1.0], [])


def test_unsupported_and_validation_errors_are_not_retried(recorded_retry_delays):
    dummy = RetryDummy(retries=3)
    attempts = []

    def unsupported():
        attempts.append('try')
        raise ChatException('unsupported', error_code='UNSUPPORTED_MEDIA_OPERATION')

    with pytest.raises(ChatException):
        dummy._retry_sync_media(unsupported, method='sync_audio_transcriptions')
    assert len(attempts) == 1

    attempts.clear()

    def invalid():
        attempts.append('try')
        raise ValueError('invalid metadata')

    with pytest.raises(ValueError):
        dummy._retry_sync_media(invalid, method='sync_audio_transcriptions')
    assert len(attempts) == 1
    assert recorded_retry_delays == ([], [])


def test_stt_retry_reuses_identical_validated_metadata(recorded_retry_delays):
    dummy = RetryDummy(retries=2)
    request = AudioTranscriptionsRequest(
        file=b'RIFF\x10\x00\x00\x00WAVEfmt ',
        model='whisper-1',
        filename='same.wav',
        content_type='audio/wav',
    )
    seen = []

    def operation():
        metadata = request.resolve_upload_metadata()
        seen.append((metadata.filename, metadata.content_type, request.transcription_fields(model='whisper-1')))
        if len(seen) == 1:
            raise HttpError('HTTP 408', status_code=408)
        return {'text': 'ok'}

    assert dummy._retry_sync_media(operation, method='sync_audio_transcriptions') == {'text': 'ok'}
    assert seen[0] == seen[1]
    assert seen[0][0] == 'same.wav'
    assert seen[0][1] == 'audio/wav'
    assert seen[0][2]['model'] == 'whisper-1'
    assert recorded_retry_delays == ([1.0], [])
