import re
import urllib.parse
from xml.sax.saxutils import escape

from magic_llm.engine.base_chat import BaseChat
from magic_llm.engine.tooling import guard_tools_supported
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest
from magic_llm.util.http import AsyncHttpClient


AZURE_TTS_OUTPUT_FORMATS = {
    'mp3': 'audio-16khz-128kbitrate-mono-mp3',
    'wav': 'riff-16khz-16bit-mono-pcm',
    'pcm': 'raw-16khz-16bit-mono-pcm',
    'ogg': 'ogg-16khz-16bit-mono-opus',
}
AZURE_STT_CONTENT_TYPES = {'audio/wav', 'audio/x-wav'}
_AZURE_LANGUAGE_RE = re.compile(r'^[a-z]{2,3}(?:-[A-Z0-9]{2,8})?$')


class EngineAzure(BaseChat):
    engine = 'azure'

    def __init__(self,
                 speech_key: str,
                 speech_region: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not speech_key:
            raise ValueError("Azure Speech requires speech_key")
        if not speech_region:
            raise ValueError("Azure Speech requires speech_region")
        self.base_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"
        self.base_transcription_url = f"https://{speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language="
        self.speech_key = speech_key

    @staticmethod
    def _validate_language(language: str | None, *, field_name: str = 'language') -> str:
        if not language or not language.strip():
            raise ValueError(f"Azure Speech requires non-empty {field_name}")
        language = language.strip()
        if not _AZURE_LANGUAGE_RE.fullmatch(language):
            raise ValueError(f"Azure Speech {field_name} must look like 'en-US' or 'es-MX'")
        return language

    @staticmethod
    def _language_from_voice(voice: str | None) -> str:
        if not voice or not voice.strip():
            raise ValueError("Azure TTS requires a non-empty voice")
        parts = voice.strip().split('-')
        if len(parts) < 2:
            raise ValueError("Azure TTS voice must include a language prefix such as 'en-US-JennyNeural'")
        return EngineAzure._validate_language('-'.join(parts[:2]), field_name='voice language')

    @staticmethod
    def _tts_output_format(response_format: str | None) -> str:
        requested = (response_format or 'mp3').strip().lower()
        if requested not in AZURE_TTS_OUTPUT_FORMATS:
            supported = ', '.join(sorted(AZURE_TTS_OUTPUT_FORMATS))
            raise ValueError(f"Unsupported Azure TTS response_format {response_format!r}. Supported: {supported}")
        return AZURE_TTS_OUTPUT_FORMATS[requested]

    def _build_ssml(self, data: AudioSpeechRequest) -> str:
        lang = self._language_from_voice(data.voice)
        voice = escape(data.voice.strip(), {'"': '&quot;', "'": '&apos;'})
        text = escape(data.input, {'"': '&quot;', "'": '&apos;'})
        return f"""
        <speak version='1.0' xml:lang='{lang}'>
            <voice xml:lang='{lang}' name='{voice}'>
                {text}
            </voice>
        </speak>
        """.strip()

    def _tts_headers(self, data: AudioSpeechRequest) -> dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self._tts_output_format(data.response_format),
            "User-Agent": "magic-audio https://arz.ai",
        }

    def _stt_headers(self, data: AudioTranscriptionsRequest) -> dict[str, str]:
        metadata = data.resolve_upload_metadata(require_metadata=True)
        content_type = metadata.content_type
        if content_type not in AZURE_STT_CONTENT_TYPES:
            raise ValueError(
                f"Azure STT supports only audio/wav content type; got {content_type}. "
                "Do not relabel non-WAV audio as WAV."
            )
        return {
            "Ocp-Apim-Subscription-Key": self.speech_key,
            "Content-Type": "audio/wav",
            "User-Agent": "magic-transcriptions https://arz.ai",
        }

    def audio_speech(self, speech_request: AudioSpeechRequest, **kwargs):
        self._unsupported_media_operation('audio_speech', 'async_audio_speech')

    def sync_audio_transcriptions(self, speech_request: AudioTranscriptionsRequest, **kwargs):
        self._unsupported_media_operation('sync_audio_transcriptions', 'async_audio_transcriptions')

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        guard_tools_supported(
            'Azure',
            kwargs.get('tools', self.kwargs.get('tools')),
            kwargs.get('tool_choice', self.kwargs.get('tool_choice')),
        )
        raise NotImplementedError

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        guard_tools_supported(
            'Azure',
            kwargs.get('tools', self.kwargs.get('tools')),
            kwargs.get('tool_choice', self.kwargs.get('tool_choice')),
        )
        raise NotImplementedError

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        guard_tools_supported(
            'Azure',
            kwargs.get('tools', self.kwargs.get('tools')),
            kwargs.get('tool_choice', self.kwargs.get('tool_choice')),
        )
        raise NotImplementedError

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        guard_tools_supported(
            'Azure',
            kwargs.get('tools', self.kwargs.get('tools')),
            kwargs.get('tool_choice', self.kwargs.get('tool_choice')),
        )
        raise NotImplementedError

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        ssml_template = self._build_ssml(data)
        headers = self._tts_headers(data)

        async def _request():
            async with AsyncHttpClient() as client:
                return await client.post_raw_binary(
                    url=self.base_url,
                    headers=headers,
                    data=ssml_template,
                    timeout=kwargs.get('timeout', 30),
                )

        return await self._retry_async_media(_request, method='async_audio_speech')

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        language = self._validate_language(data.language)
        if data.response_format not in {'json', 'verbose_json'}:
            raise ValueError("Azure STT currently supports JSON transcription responses only")
        headers = self._stt_headers(data)
        url_language = urllib.parse.quote(language, safe='-')

        async def _request():
            async with AsyncHttpClient() as client:
                return await client.post_json(
                    url=self.base_transcription_url + url_language,
                    headers=headers,
                    data=data.file,
                    timeout=kwargs.get('timeout', 30),
                )

        response = await self._retry_async_media(_request, method='async_audio_transcriptions')
        return {
            **response,
            'text': response.get('text') or response.get('DisplayText') or response.get('RecognitionStatus'),
        }
