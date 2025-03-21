import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel


class ProviderGroq(OpenAiBaseProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            **kwargs
        )

    def process_chunk(
            self, chunk: str,
            id_generation: str = '',
            last_chunk: ChatCompletionModel = None
    ) -> ChatCompletionModel:
        if chunk.startswith('data: ') and not chunk.endswith('[DONE]'):
            chunk = json.loads(chunk[5:])
            chunk['usage'] = chunk.get('x_groq', {}).get('usage', {})
            if len(chunk['choices']) == 0:
                chunk['choices'] = [{}]
            chunk = ChatCompletionModel(**chunk)
            return chunk
