# https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models

import json
import urllib.request

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChat, ModelChatResponse


class EngineOpenAI(BaseChat):
    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.base_url = base_url
        self.api_key = api_key

    def prepare_data(self, chat: ModelChat, **kwargs):
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }

        data = {
            "model": self.model,
            "messages": chat.messages,
            "stream": self.stream,
            **self.kwargs
        }

        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url + '/chat/completions', data=json_data, headers=headers)

    def prepare_data_embedding(self, text: list[str] | str, **kwargs):
        # Construct the header and data to be sent in the request.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'accept': 'application/json',
            'user-agent': 'arz-magic-llm-engine',
            **self.headers
        }

        data = {
            "input": text,
            "model": self.model,
            "encoding_format": "float",
            **kwargs
        }
        # Convert the data dictionary to a JSON string.
        json_data = json.dumps(data).encode('utf-8')

        # Create a request object with the URL, data, and headers.
        return urllib.request.Request(self.base_url + '/embeddings', data=json_data, headers=headers)

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_data(chat, **kwargs), timeout=kwargs.get('timeout')) as response:
            response_data = response.read()
            encoding = response.info().get_content_charset('utf-8')

            # Decode and print the response.
            r = json.loads(response_data.decode(encoding))
            if r['choices'][0]['message']['content']:
                return ModelChatResponse(**{
                    'content': r['choices'][0]['message']['content'],
                    'prompt_tokens': r['usage']['prompt_tokens'],
                    'completion_tokens': r['usage']['completion_tokens'],
                    'total_tokens': r['usage']['total_tokens'],
                    'role': 'assistant'
                })
            else:  # interpret as function calling
                return ModelChatResponse(**{
                    'content': r['choices'][0]['message']['tool_calls'][0]['function']['arguments'],
                    'prompt_tokens': r['usage']['prompt_tokens'],
                    'completion_tokens': r['usage']['completion_tokens'],
                    'total_tokens': r['usage']['total_tokens'],
                    'role': 'assistant'
                })

    def stream_generate(self, chat: ModelChat, **kwargs):
        # Make the request and read the response.
        with urllib.request.urlopen(self.prepare_data(chat, **kwargs), timeout=kwargs.get('timeout')) as response:
            for chunk in response:
                chunk = chunk.decode('utf-8')
                yield chunk

    def embedding(self, text: list[str] | str, **kwargs):
        with urllib.request.urlopen(self.prepare_data_embedding(text, **kwargs),
                                    timeout=kwargs.get('timeout')) as response:
            data = response.read().decode('utf-8')
            return data
