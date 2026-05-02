"""OpenAI discovery adapter.

``OpenAIDiscoveryAdapter`` is a thin subclass of
:class:`~magic_llm.engine.discovery.openai_compatible.base.OpenAICompatibleAdapter`
with ``PROVIDER = "openai"`` and ``DEFAULT_BASE_URL =
"https://api.openai.com/v1/models"``.

The block-registration pattern (one class for 13 engines) has been replaced
by per-provider concrete subclasses in ``openai_compatible/``. Only
``register_adapter("openai", OpenAIDiscoveryAdapter)`` remains here.
"""

from __future__ import annotations

from magic_llm.engine.discovery.openai_compatible.base import (
    OpenAICompatibleAdapter,
)
from magic_llm.engine.discovery import register_adapter


class OpenAIDiscoveryAdapter(OpenAICompatibleAdapter):
    """Discovery adapter for the canonical OpenAI API."""

    PROVIDER = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1/models"


register_adapter("openai", OpenAIDiscoveryAdapter)