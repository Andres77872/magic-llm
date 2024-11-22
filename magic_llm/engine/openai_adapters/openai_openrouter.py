import json
import time
import urllib

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class ProviderOpenRouter(OpenAiBaseProvider):
    def __init__(self, **kwargs):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            **kwargs
        )

    def process_chunk(
            self, chunk: str,
            id_generation: str = '',
            last_chunk: ChatCompletionModel = None
    ) -> ChatCompletionModel:
        if chunk.startswith('data: ') and not chunk.endswith('[DONE]'):
            chunk = json.loads(chunk[5:])
            chunk = ChatCompletionModel(**chunk)
            return chunk
        elif (c := chunk.strip()) and c == 'data: [DONE]':
            time.sleep(3)
            for i in range(3):
                request = urllib.request.Request(f'https://openrouter.ai/api/v1/generation?id={id_generation}',
                                                 headers=self.headers)
                with urllib.request.urlopen(request, timeout=3) as ses:
                    response = ses.read().decode('utf-8')
                    response = json.loads(response)
                    u = response['data']
                    usage = {
                        'completion_tokens': u['native_tokens_completion'],
                        'prompt_tokens': u['native_tokens_prompt'],
                        'total_tokens': u['native_tokens_prompt'] + u['native_tokens_completion']
                    }
                    last_chunk.usage = UsageModel(**usage)
                    last_chunk.choices[0].delta.content = ''
                    return last_chunk
