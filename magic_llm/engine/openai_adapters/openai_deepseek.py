import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel, PromptTokensDetailsModel


class ProviderDeepseek(OpenAiBaseProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://api.deepseek.com/v1",
            **kwargs
        )

    def process_chunk(
            self, chunk: str,
            id_generation: str = '',
            last_chunk: ChatCompletionModel = None
    ) -> ChatCompletionModel:
        if chunk.startswith('data: ') and not chunk.endswith('[DONE]'):
            chunk = json.loads(chunk[5:])
            if u := chunk.get('usage'):
                chunk['usage'] = UsageModel(prompt_tokens=u['prompt_tokens'],
                                            completion_tokens=u['completion_tokens'],
                                            total_tokens=u['total_tokens'],
                                            prompt_tokens_details=PromptTokensDetailsModel(
                                                cached_tokens=u['prompt_cache_hit_tokens']))
            if len(chunk['choices']) == 0:
                chunk['choices'] = [{}]
            chunk = ChatCompletionModel(**chunk)
            return chunk
