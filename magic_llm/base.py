from typing import Optional, Dict, Type, List, TYPE_CHECKING

from magic_llm.engine import (
    EngineOpenAI,
    EngineGoogle,
    EngineCloudFlare,
    EngineAmazon,
    EngineCohere,
    EngineAnthropic,
    EngineAzure,
)
from magic_llm.engine.base_chat import BaseChat

if TYPE_CHECKING:
    from magic_llm.engine.discovery.base_discovery import BaseDiscoveryAdapter
    from magic_llm.model.discovery import NormalizedDiscoveredModel


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

        # For Amazon, map private_key to aws_access_key_id for backward compatibility
        if engine == EngineAmazon.engine and private_key:
            engine_params['aws_access_key_id'] = private_key

        self.llm: BaseChat = engine_class(**engine_params)

    @classmethod
    def _resolve_discovery_adapter(
        cls,
        engine: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> "BaseDiscoveryAdapter":
        """Resolve and instantiate a discovery adapter for an engine.

        Single resolution point used by :meth:`BaseChat.list_models` and
        :meth:`BaseChat.async_list_models`. Adapters own their default URLs —
        passing ``base_url=None`` falls through to adapter default.

        Args:
            engine: Engine name (e.g. ``"openai"``, ``"deepinfra"``).
            api_key: API key for the provider.
            base_url: Override URL for the discovery endpoint. ``None`` uses
                      the adapter's ``DEFAULT_BASE_URL``.
            **kwargs: Additional arguments forwarded to the adapter constructor.

        Returns:
            An instantiated discovery adapter.

        Raises:
            NotImplementedError: No discovery adapter registered for *engine*.
        """
        from magic_llm.engine.discovery import get_adapter

        adapter_cls = get_adapter(engine)
        if adapter_cls is None:
            raise NotImplementedError(
                f"Engine '{engine}' has no discovery adapter registered. "
                "Model listing is not supported for this engine."
            )
        return adapter_cls(api_key=api_key, base_url=base_url, **kwargs)
