"""Perplexity discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class PerplexityDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "perplexity"
    DEFAULT_BASE_URL = "https://api.perplexity.ai/v1/models"


register_adapter("perplexity", PerplexityDiscoveryAdapter)
