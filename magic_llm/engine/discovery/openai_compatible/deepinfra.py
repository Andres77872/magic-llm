"""DeepInfra discovery adapter.

.. admonition:: URL verification
   :class: note

   DeepInfra uses ``/v1/models`` (NOT ``/v1/openai/v1/models``). The chat
   profile at ``https://api.deepinfra.com/v1/openai`` is UNRELATED to the
   discovery endpoint — do NOT reuse the chat base_url.
"""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class DeepInfraDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "deepinfra"
    DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/models"


register_adapter("deepinfra", DeepInfraDiscoveryAdapter)
