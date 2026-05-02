"""Cerebras discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class CerebrasDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "cerebras"
    DEFAULT_BASE_URL = "https://api.cerebras.ai/v1/models"


register_adapter("cerebras", CerebrasDiscoveryAdapter)
