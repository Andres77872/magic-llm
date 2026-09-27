"""Nebius AI Studio discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class NebiusDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "nebius"
    DEFAULT_BASE_URL = "https://api.studio.nebius.ai/v1/models"
    HOSTS = ('api.studio.nebius.ai', 'api.studio.nebius.com', 'api.tokenfactory.nebius.com')


register_adapter("nebius", NebiusDiscoveryAdapter)
