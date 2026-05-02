"""Novita discovery adapter.

.. admonition:: URL verification
   :class: note

   Verified via Novita docs: ``https://api.novita.ai/v3/openai/models``.
   Matches the existing docstring intent. The old code incorrectly produced
   ``/v3/openai/v1/models`` (double path).
"""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class NovitaDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "novita"
    DEFAULT_BASE_URL = "https://api.novita.ai/v3/openai/models"


register_adapter("novita", NovitaDiscoveryAdapter)
