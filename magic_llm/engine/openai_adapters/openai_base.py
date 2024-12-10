import json
import mimetypes
import aiohttp

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest, AudioTranscriptionsRequest


class ProviderOpenAI(OpenAiBaseProvider):
    def __init__(self,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs):
        super().__init__(
            base_url=base_url,
            **kwargs
        )

    def prepare_data(self, chat: ModelChat, **kwargs):
        data, headers = super().prepare_data(chat, **kwargs)
        data = json.loads(data)
        if data.get("stream"):
            data.update({
                "stream_options": {
                    "include_usage": True
                }
            })
        return json.dumps(data).encode('utf-8'), headers

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        payload = {
            **data.model_dump(),
            **kwargs
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url + '/audio/speech',
                                    headers=self.headers,
                                    json=payload) as response:
                return await response.read()

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
