"""Groq discovery adapter.

Groq uses the path ``/openai/v1/models`` (not plain ``/v1/models``).
This quirk is owned by the adapter — no central URL table.
"""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class GroqDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "groq"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1/models"


register_adapter("groq", GroqDiscoveryAdapter)
