"""Parasail discovery adapter.

.. admonition:: URL verification
   :class: note

   Verified via Parasail docs at
   ``https://docs.parasail.io/parasail-docs/serverless-and-models/serverless``:
   ``curl https://api.parasail.io/v1/models -H "Authorization: Bearer $KEY"``
"""

from magic_llm.engine.discovery import register_adapter
from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)


class ParasailDiscoveryAdapter(OpenAICompatibleAdapter):
    PROVIDER = "parasail"
    DEFAULT_BASE_URL = "https://api.parasail.io/v1/models"


register_adapter("parasail", ParasailDiscoveryAdapter)
