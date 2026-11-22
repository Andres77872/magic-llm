import pytest

from magic_llm.engine.base_chat import BaseChat
from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest


class DummyChat(BaseChat):
    engine = 'dummy-media-engine'

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        raise NotImplementedError

    def stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError

    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError


class UnlabeledDummyChat(DummyChat):
    engine = None


def _speech_request() -> AudioSpeechRequest:
    return AudioSpeechRequest(input='hello', model='tts-1', voice='alloy')


def _transcription_request() -> AudioTranscriptionsRequest:
    return AudioTranscriptionsRequest(
        file=b'RIFF\x00\x00\x00\x00WAVEdata',
        model='whisper-1',
        filename='sample.wav',
        content_type='audio/wav',
    )


def _assert_unsupported(exc: ChatException, method: str) -> None:
    assert exc.error_code == 'UNSUPPORTED_MEDIA_OPERATION'
    assert 'dummy-media-engine' in exc.message
    assert method in exc.message
    assert 'Supported alternative' in exc.message


def test_base_sync_tts_fails_loudly_not_none():
    chat = DummyChat()
    with pytest.raises(ChatException) as exc_info:
        chat.audio_speech(_speech_request())
    _assert_unsupported(exc_info.value, 'audio_speech')


async def test_base_async_tts_fails_loudly_not_none():
    chat = DummyChat()
    with pytest.raises(ChatException) as exc_info:
        await chat.async_audio_speech(_speech_request())
    _assert_unsupported(exc_info.value, 'async_audio_speech')


def test_base_sync_stt_fails_loudly_not_none():
    chat = DummyChat()
    with pytest.raises(ChatException) as exc_info:
        chat.sync_audio_transcriptions(_transcription_request())
    _assert_unsupported(exc_info.value, 'sync_audio_transcriptions')


async def test_base_async_stt_fails_loudly_not_none():
    chat = DummyChat()
    with pytest.raises(ChatException) as exc_info:
        await chat.async_audio_transcriptions(_transcription_request())
    _assert_unsupported(exc_info.value, 'async_audio_transcriptions')


def test_unsupported_media_error_uses_class_label_when_engine_missing():
    chat = UnlabeledDummyChat()
    with pytest.raises(ChatException) as exc_info:
        chat.audio_speech(_speech_request())
    assert exc_info.value.error_code == 'UNSUPPORTED_MEDIA_OPERATION'
    assert 'UnlabeledDummyChat' in exc_info.value.message
