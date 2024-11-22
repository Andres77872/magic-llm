import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model import ModelChat


class ProviderLepton(OpenAiBaseProvider):
    def __init__(self,
                 **kwargs):
        super().__init__(
            base_url=f'https://{kwargs.get("model")}.lepton.run/api/v1',
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
