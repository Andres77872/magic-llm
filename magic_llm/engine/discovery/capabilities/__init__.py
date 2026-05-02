"""Capability inference strategies for model discovery.

Provides a composable, tiered strategy system for inferring
``ModelCapabilities`` from provider API responses.

Tiers (in priority order):
    1. ProviderFieldStrategy  — read from provider API response fields
    2. ModelNameRegexStrategy — regex on model ID (consolidated patterns)
    3. ProviderDefaultsStrategy — per-provider baseline defaults
"""

from magic_llm.engine.discovery.capabilities.strategies import (
    CapabilityInferenceStrategy,
    ModelNameRegexStrategy,
    ProviderDefaultsStrategy,
    ProviderFieldStrategy,
    CompositeCapabilityInference,
)
from magic_llm.engine.discovery.capabilities.models import (
    VISION_PATTERNS,
    EMBEDDING_PATTERNS,
    FUNCTION_CALLING_PATTERNS,
    CONTEXT_WINDOW_MAP,
)

__all__ = [
    "CapabilityInferenceStrategy",
    "ModelNameRegexStrategy",
    "ProviderDefaultsStrategy",
    "ProviderFieldStrategy",
    "CompositeCapabilityInference",
    "VISION_PATTERNS",
    "EMBEDDING_PATTERNS",
    "FUNCTION_CALLING_PATTERNS",
    "CONTEXT_WINDOW_MAP",
]
