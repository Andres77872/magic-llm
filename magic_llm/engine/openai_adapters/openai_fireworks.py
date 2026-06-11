from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.exception.ChatException import ChatException
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest
from magic_llm.util.http import AsyncHttpClient


class ProviderFireworks(OpenAiBaseProvider):
    supports_stt_async = True

    ASR_HOSTS = {
        'whisper-v3': 'https://audio-prod.api.fireworks.ai/v1',
        'whisper-v3-turbo': 'https://audio-turbo.api.fireworks.ai/v1',
    }

    def __init__(self,
                 **kwargs):
        super().__init__(
            base_url="https://api.fireworks.ai/inference/v1",
            **kwargs
        )

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        headers = {
            "Authorization": self.headers.get("Authorization")
        }

        selected_model = data.model or self.model
        url = self.ASR_HOSTS.get(selected_model)
        if not url:
            supported = ', '.join(sorted(self.ASR_HOSTS))
            raise ChatException(
                message=(
                    f"Provider 'ProviderFireworks' does not support async_audio_transcriptions "
                    f"for model {selected_model!r}. Supported ASR models: {supported}."
                ),
                error_code='UNSUPPORTED_MEDIA_OPERATION',
            )

        metadata = data.resolve_upload_metadata(require_metadata=True)
        async with AsyncHttpClient() as client:
            response = await client.post_multipart(
                url=url + '/audio/transcriptions',
                fields=data.transcription_fields(model=selected_model),
                file_field='file',
                file_bytes=data.file,
                filename=metadata.filename,
                content_type=metadata.content_type,
                headers=headers,
                response_format=self._transcription_response_decode_format(data),
                timeout=kwargs.get('timeout', 30),
            )
            return self._normalize_transcription_response(response, data.response_format)
