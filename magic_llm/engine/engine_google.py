# https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=es-419

import json
import urllib.request

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineGoogle(BaseChat):
    def __init__(self,
                 api_key: str,
                 model: str,
                 stream: bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}'

        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.kwargs = kwargs

    def count_tokens(self, json_data: bytes, headers: dict) -> int:
        url_counter = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:countTokens?key={self.api_key}'
        request = urllib.request.Request(url_counter, data=json_data, headers=headers, method='POST')
        return json.loads(urllib.request.urlopen(request).read().decode('utf'))['totalTokens']

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            "contents": [
                            {
                                "role": 'user',
                                "parts": [{'text': chat.messages[0]['content']}]
                            },
                            {
                                "role": 'model',
                                "parts": [{'text': 'Ok'}]
                            }
                        ] +
                        [
                            {
                                "role": x['role'].replace('assistant', 'model'),
                                "parts": [{'text': x['content']}]
                            } for x in chat.messages[1:]],
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
            r = r['candidates'][0]['content']['parts'][0]['text']
            prompt_tokens = self.count_tokens(json_data, headers)
            data['contents'].append(
                {
                    "role": 'model',
                    "parts": [{'text': r}]
                }
            )
            json_data = json.dumps(data).encode('utf-8')
            completion_tokens = self.count_tokens(json_data, headers)

            return ModelChatResponse(**{
                'content': r,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens - prompt_tokens,
                'total_tokens': completion_tokens,
                'role': 'assistant'
            })
