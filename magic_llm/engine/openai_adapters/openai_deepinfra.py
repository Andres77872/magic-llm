import base64
import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.util.http import AsyncHttpClient
from magic_llm.util.tools_mapping import coerce_tool_choice_to_string, map_to_openai


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
            # init-time defaults first, call-time overrides second
            **self.kwargs,
            **kwargs,
        }
        if 'callback' in data:
            data.pop('callback')
        if 'fallback' in data:
            data.pop('fallback')
        # Normalize tools/tool_choice to canonical OpenAI schema
        tools_mapped, choice_mapped = map_to_openai(data.get('tools'), data.get('tool_choice'))
        if tools_mapped is not None:
            data['tools'] = tools_mapped
        if choice_mapped is not None:
            data['tool_choice'] = choice_mapped

        # DeepInfra only accepts string tool_choice. Downgrade dict to string.
        if (tc := data.get('tool_choice')) is not None:
            data['tool_choice'] = coerce_tool_choice_to_string(tc, default='auto')

        json_data = json.dumps(data).encode('utf-8')
        return json_data, self.headers

    async def async_audio_speech(self, data: AudioSpeechRequest, **kwargs):
        payload = {
            'text': data.input,
            **kwargs
        }

        if data.voice:
            payload['voice_id'] = data.voice

        url = self.base_url.replace('/openai', '') + '/inference/' + data.model
        async with AsyncHttpClient() as client:
            response = await client.post_json(url=url,
                                              json=payload,
                                              headers=self.headers)
            encoded_audio = response['audio']
            if encoded_audio and encoded_audio.startswith("data:audio/wav;base64,"):
                encoded_audio = encoded_audio.split(",")[1]
            audio_content = base64.b64decode(encoded_audio)
            return audio_content
