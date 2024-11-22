from abc import ABC
from typing import Dict, Tuple
import json

from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel


class OpenAiBaseProvider(ABC):
    def __init__(self,
                 base_url: str,
                 api_key: str,
                 model: str,
                 headers:
                 Dict[str, str] = None,
                 **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        if headers:
            self.headers = headers
        else:
            self.headers = {}
        if 'user-agent' not in self.headers:
            self.headers['user-agent'] = 'arz-magic-llm-engine'
        if 'accept' not in self.headers:
            self.headers['accept'] = 'application/json'
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'
        if 'Authorization' not in self.headers:
            self.headers['Authorization'] = f'Bearer {api_key}'

        self.kwargs = kwargs
        self.model = model

    def prepare_data(self,
                     chat:
                     ModelChat,
                     **kwargs) -> Tuple[bytes, Dict[str, str]]:
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
        json_data = json.dumps(data).encode('utf-8')
        return json_data, self.headers

    def process_chunk(self,
                      chunk: str | dict,
                      id_generation: str = '',
                      last_chunk: ChatCompletionModel = None
                      ) -> ChatCompletionModel:
        if chunk.startswith('data: ') and not chunk.endswith('[DONE]'):
            chunk = json.loads(chunk[5:])
            chunk['usage'] = c if (c := chunk.get('usage', {})) else {}
            if len(chunk['choices']) == 0:
                chunk['choices'] = [{}]
            chunk = ChatCompletionModel(**chunk)
            return chunk
