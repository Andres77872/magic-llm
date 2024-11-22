from magic_llm.engine.openai_adapters.base_provider import OpenAiBaseProvider


class ProviderFireworks(OpenAiBaseProvider):
    def __init__(self,
                 **kwargs):
        super().__init__(
            base_url="https://api.fireworks.ai/inference/v1",
            **kwargs
        )
