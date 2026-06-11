import json

from magic_llm.engine._usage_factory import build_usage_model
from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel


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
        if chunk.startswith('data: '):
            if '[DONE]' in chunk:
                return None
            chunk = json.loads(chunk[5:])
            if u := chunk.get('usage'):
                usage = build_usage_model(
                    prompt_tokens=u.get('prompt_tokens', 0),
                    completion_tokens=u.get('completion_tokens', 0),
                    total_tokens=u.get('total_tokens'),
                    cached_tokens_read=u.get('prompt_cache_hit_tokens'),
                    provider_request_id=chunk.get('id'),
                    provider_extra={
                        'prompt_cache_miss_tokens': u.get('prompt_cache_miss_tokens')
                    },
                )
                if 'prompt_cache_hit_tokens' not in u and usage.prompt_tokens_details:
                    usage.prompt_tokens_details.cached_tokens = None
                chunk['usage'] = usage
            if len(chunk['choices']) == 0:
                return None
            chunk = ChatCompletionModel(**chunk)
            return chunk
