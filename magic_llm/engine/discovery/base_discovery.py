"""Abstract base class for discovery adapters.

Per spec.md Section "Error Handling for Discovery Failures":
- DiscoveryError for provider API unreachable (5xx, timeout)
- DiscoveryRateLimitError for HTTP 429
- DiscoveryAuthError for HTTP 401
- All errors include provider name and raw response for debugging

Per Error Policy B (design.md):
- 404/not-found → return [] (graceful via DiscoveryPolicy.graceful_on_404)
- Auth (401) → propagate as DiscoveryAuthError
- Rate limit (429) → propagate as DiscoveryRateLimitError
- Server errors (5xx) → propagate as DiscoveryError
- Transport/other → propagate as DiscoveryError
- Successful empty payload → return [] (normal case, not an error)
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from magic_llm.model.discovery import NormalizedDiscoveredModel
from magic_llm.util.http import AsyncHttpClient, HttpClient, HttpError

logger = logging.getLogger(__name__)


# =============================================================================
# Discovery Exceptions (per spec.md Section "Error Handling")
# =============================================================================

class DiscoveryError(Exception):
    """Base exception for discovery failures.

    Per spec.md Section "Provider API unreachable":
    - Raised for HTTP 5xx or connection timeout
    - Includes provider name and HTTP status
    - Includes raw response body for debugging
    - Does NOT crash SDK client instance
    """

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class DiscoveryRateLimitError(DiscoveryError):
    """Exception for provider rate limiting.

    Per spec.md Section "Provider API rate limited":
    - Raised for HTTP 429 Too Many Requests
    - Includes retry-after header value if present
    - Indicates provider name
    """

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        message = f"Provider '{provider}' rate limit reached"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(
            message=message,
            provider=provider,
            status_code=429,
            response_body=response_body,
        )
        self.retry_after = retry_after


class DiscoveryAuthError(DiscoveryError):
    """Exception for authentication failures.

    Per spec.md Section "Invalid credentials":
    - Raised for HTTP 401 Unauthorized
    - Includes provider name
    - Does NOT expose attempted credentials in error message
    """

    def __init__(
        self,
        provider: str,
        response_body: Optional[str] = None,
    ):
        super().__init__(
            message=f"Provider '{provider}' credentials invalid or missing",
            provider=provider,
            status_code=401,
            response_body=response_body,
        )


class DiscoveryNotFoundError(DiscoveryError):
    """Exception for endpoint not found.

    Used for providers where listing endpoint doesn't exist.
    Under Error Policy B, this is only raised if graceful_on_404=False.
    """

    def __init__(
        self,
        provider: str,
        endpoint: str,
        response_body: Optional[str] = None,
    ):
        super().__init__(
            message=f"Provider '{provider}' listing endpoint '{endpoint}' not found",
            provider=provider,
            status_code=404,
            response_body=response_body,
        )


# =============================================================================
# Discovery Policy (Error Policy B)
# =============================================================================


@dataclass
class DiscoveryPolicy:
    """Controls error handling behavior for a discovery adapter.

    Default implements Error Policy B:
    - 404/not-found → return [] (graceful)
    - Auth (401) → propagate as DiscoveryAuthError
    - Rate limit (429) → propagate as DiscoveryRateLimitError
    - Server errors (5xx) → propagate as DiscoveryError
    - Transport/other → propagate as DiscoveryError
    """

    graceful_on_404: bool = True     # Error Policy B: 404 → return []
    graceful_on_5xx: bool = False    # Error Policy B: always propagate 5xx
    graceful_on_other: bool = False  # Reserved for future use
    warn_on_graceful: bool = True    # Log warning when degrading


# =============================================================================
# Abstract Discovery Adapter
# =============================================================================


class BaseDiscoveryAdapter(abc.ABC):
    """Abstract base class for provider model discovery.

    Per spec.md Section "Optional Discovery Interface":
    - Provides discover() and async_discover() methods
    - Returns List[NormalizedDiscoveredModel]
    - Handles provider-specific auth patterns
    - Maps HTTP errors to discovery-specific exceptions

    Subclasses implement:
    - _get_endpoint_url(): Provider-specific listing endpoint
    - _get_headers(): Provider-specific auth headers
    - Pipeline overrides as needed (_extract_raw_models, _infer_capabilities, etc.)
    """

    # Error handling policy — default is Error Policy B
    _discovery_policy: DiscoveryPolicy = DiscoveryPolicy()

    def __init__(
        self,
        provider: str,
        base_url: str,
        **kwargs
    ):
        """Initialize discovery adapter.

        Args:
            provider: Engine identifier (openai, anthropic, etc.)
            base_url: Provider API base URL
            **kwargs: Additional credentials (api_key, etc.)
        """
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.kwargs = kwargs

    # ── Credential Resolution (Phase 4) ───────────────────────────────────

    @classmethod
    def resolve_credentials(cls, chat_instance) -> Dict[str, Optional[str]]:
        """Extract api_key and base_url from a chat instance.

        Resolution chain:
        1. ``api_key``: ``self.api_key`` → ``self.base.api_key`` → ``None``
           (empty string normalised to ``None``)
        2. ``base_url``: ``self.base_url`` → ``None``

        Returns:
            Dict with ``api_key`` (str | None) and ``base_url`` (str | None).
        """
        api_key = (
            getattr(chat_instance, "api_key", None)
            or getattr(getattr(chat_instance, "base", None), "api_key", None)
        )
        # Fix: empty string → None
        if api_key == "" or api_key is None:
            api_key = None
        return {
            "api_key": api_key,
            "base_url": getattr(chat_instance, "base_url", None),
        }

    # ── Abstract methods ──────────────────────────────────────────────────

    @abc.abstractmethod
    def _get_endpoint_url(self) -> str:
        """Get the provider-specific listing endpoint URL.

        Returns:
            Full URL for model listing endpoint
        """
        pass

    @abc.abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """Get provider-specific auth headers.

        Returns:
            Headers dict for HTTP request
        """
        pass

    # ── Normalization Pipeline (overridable steps) ────────────────────────

    def _normalize_response(self, raw_response: Dict[str, Any]) -> List[NormalizedDiscoveredModel]:
        """Normalize provider-specific response to standard contract.

        Concrete implementation: extracts raw model list via
        ``_extract_raw_models()``, normalises each via
        ``_normalize_single_model()``.

        Args:
            raw_response: Raw JSON response from provider API

        Returns:
            List of NormalizedDiscoveredModel objects
        """
        raw_models = self._extract_raw_models(raw_response)
        if not raw_models:
            return []
        return [self._normalize_single_model(m) for m in raw_models]

    def _extract_raw_models(self, raw_response: Dict[str, Any] | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract the list of model records from the provider response shape.

        Default: ``raw_response.get("data", [])`` or ``raw_response.get("models", [])``.
        Override for providers with non-standard response shapes (e.g. Azure Foundry 'value').

        Safety net: bare JSON arrays are returned as-is to avoid ``AttributeError``
        when a misconfigured engine string causes the wrong adapter to be selected.
        """
        if isinstance(raw_response, list):
            return raw_response
        return raw_response.get("data", []) or raw_response.get("models", [])

    def _normalize_single_model(self, raw_model: Dict[str, Any]) -> NormalizedDiscoveredModel:
        """Normalise a single raw model record into the standard contract."""
        model_id = self._extract_model_id(raw_model)
        return NormalizedDiscoveredModel(
            external_id=model_id,
            provider=self.provider,
            display_name=self._extract_display_name(raw_model),
            description=self._extract_description(raw_model),
            capabilities=self._infer_capabilities(raw_model),
            context_window=self._infer_context_window(raw_model),
            max_output_tokens=self._infer_max_output_tokens(raw_model),
            pricing=self._extract_pricing(raw_model),
            raw=raw_model,
        )

    def _extract_model_id(self, raw_model: Dict[str, Any]) -> str:
        """Extract the canonical model ID from a raw model record."""
        return raw_model.get("id", raw_model.get("name", ""))

    def _extract_display_name(self, raw_model: Dict[str, Any]) -> Optional[str]:
        """Extract a human-readable display name."""
        return raw_model.get("display_name") or raw_model.get("name") or raw_model.get("id")

    def _extract_description(self, raw_model: Dict[str, Any]) -> Optional[str]:
        """Extract the provider description."""
        return raw_model.get("description")

    def _infer_capabilities(self, raw_model: Dict[str, Any]):
        """Infer model capabilities via the configured strategy."""
        from magic_llm.engine.discovery.capabilities import (
            CompositeCapabilityInference,
            ProviderDefaultsStrategy,
        )
        model_id = self._extract_model_id(raw_model)
        strategy = getattr(self, "_capability_strategy", None)
        if strategy is None:
            strategy = CompositeCapabilityInference([ProviderDefaultsStrategy()])
        return strategy.infer(self.provider, model_id, raw_model)

    def _infer_context_window(self, raw_model: Dict[str, Any]) -> Optional[int]:
        """Infer maximum context window."""
        return raw_model.get("context_window") or raw_model.get("context_length")

    def _infer_max_output_tokens(self, raw_model: Dict[str, Any]) -> Optional[int]:
        """Infer maximum output tokens."""
        return raw_model.get("max_output_tokens") or raw_model.get("max_tokens")

    def _extract_pricing(self, raw_model: Dict[str, Any]) -> Optional[Any]:
        """Extract pricing info.  Only OpenRouter and Azure Foundry override this."""
        return None

    # ── Error handling ────────────────────────────────────────────────────

    def _handle_http_error(self, error: HttpError) -> None:
        """Convert HTTP errors to discovery-specific exceptions.

        Under Error Policy B (``DiscoveryPolicy`` default):
        - 401 → raises ``DiscoveryAuthError`` (always)
        - 429 → raises ``DiscoveryRateLimitError`` (always)
        - 404 → if ``graceful_on_404``: logs warning, returns ``None``;
                else raises ``DiscoveryNotFoundError``
        - 5xx → if ``graceful_on_5xx``: logs warning, returns ``None``;
                else raises ``DiscoveryError``
        - Other → if ``graceful_on_other``: logs warning, returns ``None``;
                  else raises ``DiscoveryError``

        When this method returns ``None`` (graceful degradation),
        ``discover()`` / ``async_discover()`` returns ``[]``.

        Args:
            error: HttpError from HTTP client

        Raises:
            DiscoveryRateLimitError: HTTP 429
            DiscoveryAuthError: HTTP 401
            DiscoveryNotFoundError: HTTP 404 with graceful_on_404=False
            DiscoveryError: HTTP 5xx or other errors (default)
        """
        policy = self._discovery_policy
        response_body = None
        if error.response_content:
            try:
                response_body = error.response_content.decode('utf-8', errors='replace')
            except Exception:
                response_body = str(error.response_content)

        if error.status_code == 429:
            # ALWAYS propagate — no graceful flag for rate limits
            retry_after = None
            if response_body:
                try:
                    data = json.loads(response_body)
                    retry_after = data.get('retry_after') or data.get('retry-after')
                except json.JSONDecodeError:
                    pass
            raise DiscoveryRateLimitError(
                provider=self.provider,
                retry_after=retry_after,
                response_body=response_body,
            )

        if error.status_code == 401:
            # ALWAYS propagate — no graceful flag for auth errors
            raise DiscoveryAuthError(
                provider=self.provider,
                response_body=response_body,
            )

        if error.status_code == 404:
            if policy.graceful_on_404:
                if policy.warn_on_graceful:
                    logger.warning(
                        "[%s] %s returned 404 — returning empty list. "
                        "Endpoint: %s",
                        self.provider,
                        self._get_endpoint_url(),
                        response_body or "",
                    )
                return None
            raise DiscoveryNotFoundError(
                provider=self.provider,
                endpoint=self._get_endpoint_url(),
                response_body=response_body,
            )

        if 500 <= error.status_code < 600:
            if policy.graceful_on_5xx:
                if policy.warn_on_graceful:
                    logger.warning(
                        "[%s] %s returned HTTP %d — returning empty list.",
                        self.provider,
                        self._get_endpoint_url(),
                        error.status_code,
                    )
                return None
            raise DiscoveryError(
                message=f"Provider '{self.provider}' discovery failed: HTTP {error.status_code}",
                provider=self.provider,
                status_code=error.status_code,
                response_body=response_body,
            )

        # Other errors (timeout, connection, etc.)
        if policy.graceful_on_other:
            if policy.warn_on_graceful:
                logger.warning(
                    "[%s] Discovery error (HTTP %d) — returning empty list.",
                    self.provider,
                    error.status_code,
                )
            return None
        raise DiscoveryError(
            message=f"Provider '{self.provider}' discovery failed: HTTP {error.status_code}",
            provider=self.provider,
            status_code=error.status_code,
            response_body=response_body,
        )

    def _parse_bytes(self, response_bytes: bytes) -> List[NormalizedDiscoveredModel]:
        """Parse raw response bytes into normalized models.

        Shared by sync ``discover()`` and async ``async_discover()`` to
        avoid duplicating JSON-parse and error-handling logic (~25 lines
        removed from each transport path).

        Args:
            response_bytes: Raw HTTP response body.

        Returns:
            List of NormalizedDiscoveredModel objects.

        Raises:
            DiscoveryError: Invalid JSON in response body.
        """
        try:
            raw_response = json.loads(response_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            # Include a preview of the actual response body so callers can
            # diagnose what the provider returned (HTML error page, plain
            # text, redirect body, etc.) instead of a blind "invalid JSON".
            preview = response_bytes[:200].decode('utf-8', errors='replace').strip()
            preview = ' '.join(preview.split())  # collapse whitespace
            raise DiscoveryError(
                message=(
                    f"Provider '{self.provider}' returned invalid JSON "
                    f"({e.msg} at pos {e.pos}). Response preview: {preview!r}"
                ),
                provider=self.provider,
                response_body=response_bytes[:1024].decode('utf-8', errors='replace'),
            )
        return self._normalize_response(raw_response)

    # ── Sync discover() template ──────────────────────────────────────────

    def discover(self) -> List[NormalizedDiscoveredModel]:
        """Synchronous discovery of provider models.

        Returns:
            List of NormalizedDiscoveredModel objects

        Raises:
            DiscoveryError: Provider API unreachable
            DiscoveryRateLimitError: Rate limited
            DiscoveryAuthError: Invalid credentials
        """
        endpoint = self._get_endpoint_url()
        headers = self._get_headers()

        try:
            with HttpClient() as client:
                response_bytes = client.request("GET", endpoint, headers=headers)
            return self._parse_bytes(response_bytes)

        except HttpError as e:
            # _handle_http_error may raise (propagate) or return None (graceful)
            self._handle_http_error(e)
            return []

    # ── Async discover() template ─────────────────────────────────────────

    async def async_discover(self) -> List[NormalizedDiscoveredModel]:
        """Asynchronous discovery of provider models.

        Returns:
            List of NormalizedDiscoveredModel objects

        Raises:
            DiscoveryError: Provider API unreachable
            DiscoveryRateLimitError: Rate limited
            DiscoveryAuthError: Invalid credentials
        """
        endpoint = self._get_endpoint_url()
        headers = self._get_headers()

        logger.debug(
            "[%s] async discovery GET %s", self.provider, endpoint
        )

        try:
            async with AsyncHttpClient() as client:
                response_bytes = await client.request("GET", endpoint, headers=headers)
            return self._parse_bytes(response_bytes)

        except HttpError as e:
            # _handle_http_error may raise (propagate) or return None (graceful)
            self._handle_http_error(e)
            return []
