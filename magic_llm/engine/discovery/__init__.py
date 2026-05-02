"""Discovery module for provider model catalog listing.

Per spec.md Section "SDK Discovery Module":
- Provides optional list_models() methods on BaseChat
- Per-engine discovery adapters for all supported providers
- Normalization layer returning NormalizedDiscoveredModel contract

Each provider owns its discovery URL via a concrete adapter subclass.
No central URL table, no base_url normalization.
"""

from __future__ import annotations

from typing import Dict, Type, Optional, List, Any

from magic_llm.model.discovery import NormalizedDiscoveredModel
from magic_llm.engine.discovery.base_discovery import (
    BaseDiscoveryAdapter,
    DiscoveryError,
    DiscoveryRateLimitError,
    DiscoveryAuthError,
    DiscoveryNotFoundError,
)


# Adapter registry: maps engine types to discovery adapters
# Populated by adapters on import
_ADAPTER_REGISTRY: Dict[str, Type[BaseDiscoveryAdapter]] = {}


def register_adapter(engine: str, adapter_class: Type[BaseDiscoveryAdapter]) -> None:
    """Register a discovery adapter for an engine type.
    
    Args:
        engine: Engine identifier (openai, anthropic, etc.)
        adapter_class: Discovery adapter class
    """
    _ADAPTER_REGISTRY[engine] = adapter_class


def get_adapter(engine: str) -> Optional[Type[BaseDiscoveryAdapter]]:
    """Get the discovery adapter for an engine type.
    
    Args:
        engine: Engine identifier
        
    Returns:
        Adapter class if registered, None otherwise
    """
    return _ADAPTER_REGISTRY.get(engine)


def supports_discovery(engine: str) -> bool:
    """Check if an engine supports model discovery.
    
    Args:
        engine: Engine identifier
        
    Returns:
        True if adapter registered, False otherwise
    """
    return engine in _ADAPTER_REGISTRY


def list_supported_engines() -> List[str]:
    """Return the list of engine names that have a registered discovery adapter.
    
    This is the public surface for downstream consumers (e.g. ``api.magic_llm``)
    to enumerate supported discovery engines without reaching into the private
    ``_ADAPTER_REGISTRY``.
    
    Returns:
        Sorted list of engine names with registered discovery adapters.
    """
    return sorted(_ADAPTER_REGISTRY.keys())


# Engine types supported for discovery in v1
# Each provider now has its own concrete adapter — see openai_compatible/ and
# individual adapter files for registration calls.
DISCOVERY_SUPPORTED_ENGINES: List[str] = [
    # OpenAI-compatible (own adapter per provider)
    "openai",
    "deepinfra",
    "groq",
    "novita",
    "perplexity",
    "together",
    "mistral",
    "deepseek",
    "hyperbolic",
    "cerebras",
    "xai",
    "parasail",
    "nebius",
    # Custom-endpoint providers
    "anthropic",
    "cohere",
    "google",
    # OpenAI-compatible variants
    "sambanova",
    "openrouter",
    # Azure surfaces
    "azure",
    "azure-foundry",
]

# Engines known to NOT have a model listing API
# These return None from get_adapter()
DISCOVERY_UNSUPPORTED_ENGINES: List[str] = [
    "amazon",
    "cloudflare",
    "azure-speech",
]


__all__ = [
    # Base classes
    "BaseDiscoveryAdapter",
    "DiscoveryError",
    "DiscoveryRateLimitError",
    "DiscoveryAuthError",
    "DiscoveryNotFoundError",
    # Registry functions
    "register_adapter",
    "get_adapter",
    "supports_discovery",
    "list_supported_engines",
    # Constants
    "DISCOVERY_SUPPORTED_ENGINES",
    "DISCOVERY_UNSUPPORTED_ENGINES",
    # Model
    "NormalizedDiscoveredModel",
]


# Import the openai_compatible subpackage so all concrete adapters self-register
from magic_llm.engine.discovery import openai_compatible  # noqa: F401, E402

# Import openai_discovery so register_adapter("openai", OpenAIDiscoveryAdapter) fires
from magic_llm.engine.discovery import openai_discovery  # noqa: F401, E402

# Import remaining (non-OpenAI-compatible) adapters
from magic_llm.engine.discovery.anthropic_discovery import AnthropicDiscoveryAdapter  # noqa: F401, E402
from magic_llm.engine.discovery.cohere_discovery import CohereDiscoveryAdapter  # noqa: F401, E402
from magic_llm.engine.discovery.google_discovery import GoogleDiscoveryAdapter  # noqa: F401, E402
from magic_llm.engine.discovery.sambanova_discovery import SambaNovaDiscoveryAdapter  # noqa: F401, E402
from magic_llm.engine.discovery.openrouter_discovery import OpenRouterDiscoveryAdapter  # noqa: F401, E402
from magic_llm.engine.discovery.azure_discovery import (  # noqa: F401, E402
    AzureOpenAIDiscoveryAdapter,
    AzureFoundryDiscoveryAdapter,
)
