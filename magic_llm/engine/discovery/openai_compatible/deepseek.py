"""DeepSeek discovery adapter."""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class DeepSeekDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1/models"


register_adapter("deepseek", DeepSeekDiscoveryAdapter)
