# https://developers.cloudflare.com/workers-ai/models/text-generation

import json
import urllib.request

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineCloudFlare(BaseChat):
    def __init__(self,
                 api_key: str,
                 model: str,
                 account_id: str,
                 stream: bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}'

        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.kwargs = kwargs

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Construct the header and data to be sent in the request.
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        data = {
            "messages": chat.messages,
            **kwargs
        }

        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        request = urllib.request.Request(self.url, data=json_data, headers=headers, method='POST')

        # Make the request and read the response.
        with urllib.request.urlopen(request) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')

            # Decode the response.
            r = json.loads(response_data.decode(encoding))

            r = r['result']['response']

            return ModelChatResponse(**{
                'content': r,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'role': 'assistant'
            })

    def stram_generate(self, chat: ModelChat, **kwargs):
        raise NotImplementedError
