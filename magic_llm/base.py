from magic_llm.engine import (EngineOpenAI,
                              EngineGoogle,
                              EngineCloudFlare,
                              EngineAmazon,
                              EngineCohere)


class MagicLlmBase:
    def __init__(self,
                 engine: str,
                 model: str,
                 private_key: str | None,
                 **kwargs):
        self.private_key = private_key
        if engine == 'openai':
            self.llm = EngineOpenAI(
                api_key=private_key,
                model=model,
                **kwargs
            )
        elif engine == 'google':
            self.llm = EngineGoogle(
                api_key=private_key,
                model=model,
                **kwargs
            )
        elif engine == 'cloudflare':
            self.llm = EngineCloudFlare(
                api_key=private_key,
                model=model,
                **kwargs
            )
        elif engine == 'amazon':
            self.llm = EngineAmazon(
                model=model,
                **kwargs
            )
        elif engine == 'cohere':
            self.llm = EngineCohere(
                model=model,
                api_key=private_key,
                **kwargs
            )
