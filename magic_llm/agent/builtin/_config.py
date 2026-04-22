"""Internal configuration helpers for builtin browsing tools.

This module provides provider resolution logic used by browsing.py tools.
Separated to avoid circular imports and keep configuration centralized.
"""

import os
from typing import Optional


# Environment variable for default provider override
_BROWSING_DEFAULT_PROVIDER_ENV = "BROWSING_DEFAULT_PROVIDER"


def get_default_provider() -> str:
    """Get the default browsing provider from environment or fallback.

    Returns:
        Default provider name ("serper", "tavily", or "exa").

    The provider is resolved from BROWSING_DEFAULT_PROVIDER environment
    variable if set, otherwise defaults to "serper" (the primary provider).
    """
    env_provider = os.environ.get(_BROWSING_DEFAULT_PROVIDER_ENV)
    if env_provider:
        # Normalize to lowercase
        normalized = env_provider.lower().strip()
        valid_providers = ("serper", "tavily", "exa")
        if normalized in valid_providers:
            return normalized
        # Invalid env value: fall back to default silently
        # (validation happens at request level, not here)
    return "serper"


def get_request_provider(
    request_provider: Optional[str],
    default_provider: Optional[str] = None
) -> str:
    """Resolve the provider for a specific request.

    Args:
        request_provider: Provider from request features.browsing.provider.
        default_provider: Override default (if None, uses get_default_provider()).

    Returns:
        Resolved provider name.

    Priority:
    1. request_provider if provided and valid
    2. default_provider if provided
    3. get_default_provider() fallback
    """
    if request_provider:
        return request_provider.lower()
    if default_provider:
        return default_provider.lower()
    return get_default_provider()


__all__ = ["get_default_provider", "get_request_provider"]