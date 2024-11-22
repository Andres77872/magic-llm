from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider


class ProviderMistral(OpenAiBaseProvider):
    def __init__(self,
                 **kwargs):
        super().__init__(
            base_url="https://api.mistral.ai/v1",
            **kwargs
        )
