# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models

import json
import urllib.request

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineOpenAI(BaseChat):
    def __init__(self,
                 api_key: str,
                 model: str,
                 stream: bool = False,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs) -> None:
        super().__init__()
        self.base_url = base_url + '/chat/completions'
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.kwargs = kwargs

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        data = {
            "model": self.model,
            "messages": chat.messages,
            **kwargs
        }

        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        request = urllib.request.Request(self.base_url, data=json_data, headers=headers)

        # Make the request and read the response.
        with urllib.request.urlopen(request) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')

            # Decode and print the response.
            r = json.loads(response_data.decode(encoding))
            return ModelChatResponse(**{
                'content': r['choices'][0]['message']['content'],
                'prompt_tokens': r['usage']['prompt_tokens'],
                'completion_tokens': r['usage']['completion_tokens'],
                'total_tokens': r['usage']['total_tokens'],
                'role': 'assistant'
            })
