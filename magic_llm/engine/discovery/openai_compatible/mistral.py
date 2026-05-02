"""Mistral discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class MistralDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "mistral"
    DEFAULT_BASE_URL = "https://api.mistral.ai/v1/models"


register_adapter("mistral", MistralDiscoveryAdapter)
