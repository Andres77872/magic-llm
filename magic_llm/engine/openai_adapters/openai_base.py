import json
import urllib.parse

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.util.http import AsyncHttpClient


def _is_official_openai_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and parsed.hostname.lower() == "api.openai.com":
        path = parsed.path.rstrip("/")
        if path in ("", "/v1"):
            return True
    return False


class ProviderOpenAI(OpenAiBaseProvider):
    def __init__(self,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs):
        super().__init__(
            base_url=base_url,
            **kwargs
        )

    def transform_request(self, chat: ModelChat, **kwargs):
        json_data, headers = super().transform_request(chat, **kwargs)
        data = json.loads(json_data)
        if _is_official_openai_url(self.base_url):
            if "max_tokens" in data:
                if "max_completion_tokens" not in data:
                    data["max_completion_tokens"] = data.pop("max_tokens")
                else:
                    del data["max_tokens"]
        if data.get("stream"):
            data["stream_options"] = {"include_usage": True}
        return json.dumps(data).encode("utf-8"), headers

    def prepare_data(self, chat: ModelChat, **kwargs):
        return self.transform_request(chat, **kwargs)

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
