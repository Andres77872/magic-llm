"""Together AI discovery adapter.

Together's ``/v1/models`` endpoint returns a **bare JSON array** rather than
the OpenAI-standard ``{"data": [...]}`` wrapper.  We override
``_extract_raw_models`` to handle this non-standard shape.
"""

from typing import Any, Dict, List

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class TogetherDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "together"
    DEFAULT_BASE_URL = "https://api.together.xyz/v1/models"

    def _extract_raw_models(
        self, raw_response: Dict[str, Any] | List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Override: Together returns a bare JSON array, not ``{"data": [...]}``."""
        if isinstance(raw_response, list):
            return raw_response
        return super()._extract_raw_models(raw_response)


register_adapter("together", TogetherDiscoveryAdapter)
