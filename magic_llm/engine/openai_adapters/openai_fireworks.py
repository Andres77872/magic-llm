from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest
from magic_llm.util.http import AsyncHttpClient


class ProviderFireworks(OpenAiBaseProvider):
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

        if data.model == 'whisper-v3':
            url = 'https://audio-prod.us-virginia-1.direct.fireworks.ai/v1'
        elif data.model == 'whisper-v3-turbo':
            url = 'https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1'
        else:
            raise NotImplementedError
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=url + '/audio/transcriptions',
                                              data=self.prepare_transcriptions(data),
                                              headers=headers)
            return response
