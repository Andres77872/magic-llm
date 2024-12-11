import base64

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.util.http import AsyncHttpClient


class ProviderDeepInfra(OpenAiBaseProvider):
    def __init__(self,
                 base_url: str = "https://api.deepinfra.com/v1/openai",
                 **kwargs):
        super().__init__(
            base_url=base_url,
            **kwargs
        )

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        payload = {
            'preset_voice': data.voice,
            'text': data.input,
            'output_format': 'wav',
            'speed': data.speed,
            **kwargs
        }
        url = self.base_url.replace('/openai', '') + '/inference/deepinfra/tts'
        async with AsyncHttpClient() as client:
            response = await client.post_raw_binary(url=url,
                                                    json=payload,
                                                    headers=self.headers)
            encoded_audio = response.get("audio")
            if encoded_audio and encoded_audio.startswith("data:audio/wav;base64,"):
                encoded_audio = encoded_audio.split(",")[1]
            audio_content = base64.b64decode(encoded_audio)
            return audio_content
