import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.util.http import AsyncHttpClient


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

        async with AsyncHttpClient() as client:
            response = await client.post_raw_binary(
                url=self.base_url + '/audio/speech',
                json=payload,
                headers=self.headers)
            return response
