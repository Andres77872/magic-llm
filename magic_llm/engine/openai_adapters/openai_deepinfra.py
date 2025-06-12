import base64
import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat
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

    def prepare_data(self,
                     chat:
                     ModelChat,
                     **kwargs) -> tuple[bytes, dict[str, str]]:
        # Construct the header and data to be sent in the request.
        messages = chat.get_messages()
        for message in messages:
            if message['role'] == 'user' and isinstance(message['content'], list):
                for item in message['content']:
                    if item.get('type') == 'text':
                        item.pop('image_url', None)
                    if item.get('type') == 'image_url':
                        item.pop('text', None)
        data = {
            "model": self.model,
            "messages": messages,
            **kwargs,
            **self.kwargs
        }
        if 'callback' in data:
            data.pop('callback')
        if 'fallback' in data:
            data.pop('fallback')
        if (tc := data.get('tool_choice')) and type(tc) == dict:
            data['tool_choice'] = 'auto'

        json_data = json.dumps(data).encode('utf-8')
        return json_data, self.headers

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
