"""Hyperbolic discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class HyperbolicDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "hyperbolic"
    DEFAULT_BASE_URL = "https://api.hyperbolic.xyz/v1/models"


register_adapter("hyperbolic", HyperbolicDiscoveryAdapter)
