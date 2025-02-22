from typing import Optional, Dict, Type
from magic_llm.engine import (
    EngineOpenAI,
    EngineGoogle,
    EngineCloudFlare,
    EngineAmazon,
    EngineCohere,
    EngineAnthropic,
    EngineAzure,
)


class MagicLlmBase:
    # Define engine mapping as a class attribute
    ENGINE_MAP: Dict[str, Type] = {
        EngineOpenAI.engine: EngineOpenAI,
        EngineGoogle.engine: EngineGoogle,
        EngineCloudFlare.engine: EngineCloudFlare,
        EngineAmazon.engine: EngineAmazon,
        EngineCohere.engine: EngineCohere,
        EngineAnthropic.engine: EngineAnthropic,
        EngineAzure.engine: EngineAzure,
    }

    def __init__(
            self,
            engine: str,
            private_key: Optional[str] = None,
            model: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Initialize the MagicLlmBase instance.

        Args:
            engine: The name of the LLM engine to use
            private_key: API key for authentication
            model: The specific model to use
            **kwargs: Additional arguments for the engine initialization

        Raises:
            ValueError: If an unsupported engine is specified
        """
        self.private_key = private_key

        engine_class = self.ENGINE_MAP.get(engine)
        if not engine_class:
            raise ValueError(f"Unsupported engine: {engine}")

        # Prepare engine initialization parameters
        engine_params = kwargs.copy()
        if model:
            engine_params['model'] = model

        # Add API key for engines that require it
        if engine not in {EngineAmazon.engine, EngineAzure.engine}:
            engine_params['api_key'] = private_key

        self.llm = engine_class(**engine_params)
