import json

from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider
from magic_llm.model.ModelChatStream import ChatCompletionModel


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
    ) -> ChatCompletionModel | None:
        """Pure SSE-to-model transformation — no HTTP calls, no side effects.

        Usage polling is handled at the engine level (EngineOpenAI.stream_generate
        and async_stream_generate) after the stream loop completes.
        """
        if chunk.startswith('data: ') and '[DONE]' not in chunk:
            chunk = json.loads(chunk[5:])
            if len(chunk.get('choices', [])) == 0:
                return None
            return ChatCompletionModel(**chunk)
        elif (c := chunk.strip()) and c == 'data: [DONE]':
            # Signal completion — usage polling is done by the engine
            return None
