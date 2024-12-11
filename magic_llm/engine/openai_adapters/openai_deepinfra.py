import mimetypes

import aiohttp
import base64

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest


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
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url.replace('/openai', '') + '/inference/deepinfra/tts',
                                    headers=self.headers,
                                    json=payload) as response:
                response_json = await response.json()
                encoded_audio = response_json.get("audio")
                if encoded_audio and encoded_audio.startswith("data:audio/wav;base64,"):
                    encoded_audio = encoded_audio.split(",")[1]
                audio_content = base64.b64decode(encoded_audio)
                return audio_content

    async def async_audio_transcriptions(self, data: AudioTranscriptionsRequest, **kwargs):
        headers = {
            "Authorization": self.headers.get("Authorization")
        }
        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            data.file,
            filename="audio.mp3",
            content_type=mimetypes.guess_type("audio.mp3")[0] or "application/octet-stream"
        )
        form_data.add_field('model', data.model)
        if data.language:
            form_data.add_field('language', data.language)
        if data.prompt:
            form_data.add_field('prompt', data.prompt)
        if data.response_format:
            form_data.add_field('response_format', data.response_format)
        if data.temperature is not None:
            form_data.add_field('temperature', str(data.temperature))

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.base_url + '/audio/transcriptions',
                    headers=headers,
                    data=form_data
            ) as response:
                response.raise_for_status()
                return await response.json()
