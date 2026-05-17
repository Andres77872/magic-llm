import base64
import json
import logging
import os

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider, _dump_payload, _dump_payload_full

logger = logging.getLogger(__name__)
from magic_llm.engine.tooling import map_request_tools
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

        # Strip non-standard is_error field from tool messages.
        # is_error is stored in ModelChat for internal debugging/tracing but is NOT
        # part of the OpenAI-compatible spec — strict providers may reject it.
        messages = [
            {k: v for k, v in msg.items() if k != 'is_error'}
            if msg.get('role') == 'tool' else msg
            for msg in messages
        ]

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
        # Normalize through the engine/core tooling boundary and preserve named
        # forced tool choices instead of silently downgrading them to "auto".
        request_tools = map_request_tools('deepinfra', data.get('tools'), data.get('tool_choice'))
        if request_tools.tools is not None:
            data['tools'] = request_tools.tools
        elif 'tools' in data:
            data.pop('tools')
        if request_tools.tool_choice is not None:
            data['tool_choice'] = request_tools.tool_choice
        elif 'tool_choice' in data:
            data.pop('tool_choice')

        if os.environ.get("MAGIC_LLM_DEBUG_PAYLOAD"):
            logger.info("MAGIC_LLM_DEBUG_PAYLOAD %s", _dump_payload(self, data))

        if os.environ.get("MAGIC_LLM_DEBUG_PAYLOAD_FULL"):
            logger.info("MAGIC_LLM_DEBUG_PAYLOAD_FULL %s", _dump_payload_full(self, data))

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
