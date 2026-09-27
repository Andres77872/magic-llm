"""Perplexity discovery adapter.

Perplexity serves chat completions at the API root
(``https://api.perplexity.ai/chat/completions``), so providers store the bare
host as their base URL. The model listing, however, only exists under the
versioned path ``/v1/models`` — ``https://api.perplexity.ai/models`` is a 404.
"""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.base_discovery import versioned_endpoint
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class PerplexityDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "perplexity"
    DEFAULT_BASE_URL = "https://api.perplexity.ai/v1/models"
    HOSTS = ('api.perplexity.ai',)

    def _get_endpoint_url(self) -> str:
        return versioned_endpoint(self.base_url, "v1")


register_adapter("perplexity", PerplexityDiscoveryAdapter)
