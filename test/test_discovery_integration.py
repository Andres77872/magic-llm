"""Integration tests for the discovery layer.

Covers:
- End-to-end via ``MagicLLM(engine=...).list_models()`` with mocked HTTP
- Registry completeness — every expected name resolves via ``get_adapter()``
- ``get_adapter("azure-speech")`` returns ``None``
- ``get_adapter("definitely-not-real")`` returns ``None``
- Sync and async produce identical normalized output for the same payload
- Backward-compatible import path for ``get_adapter``
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm import MagicLLM
from magic_llm.engine.discovery import get_adapter, list_supported_engines
from magic_llm.engine.discovery.base_discovery import (
    BaseDiscoveryAdapter,
    DiscoveryError,
    DiscoveryAuthError,
    DiscoveryRateLimitError,
)
from magic_llm.engine.discovery.openai_discovery import (
    OpenAIDiscoveryAdapter,
)
from magic_llm.util.http import HttpError


# ── Shared test payload ──────────────────────────────────────────────────

OPENAI_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai"},
        {"id": "gpt-4-turbo", "object": "model", "created": 1700000001, "owned_by": "openai"},
    ],
}


# ── End-to-end via MagicLLM ─────────────────────────────────────────────

class TestMagicLLMEndToEnd:
    """Full end-to-end: MagicLLM → BaseChat → adapter → mocked HTTP."""

    @pytest.fixture
    def mock_http(self):
        """Patch both HttpClient and AsyncHttpClient."""
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as sync_mock, patch(
            "magic_llm.engine.discovery.base_discovery.AsyncHttpClient"
        ) as async_mock:
            # Sync mock
            sync_instance = MagicMock()
            sync_instance.request.return_value = json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            sync_instance.__enter__.return_value = sync_instance
            sync_mock.return_value = sync_instance
            # Async mock
            async_instance = MagicMock()
            async_instance.request = AsyncMock(
                return_value=json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            )
            async_instance.__aenter__.return_value = async_instance
            async_mock.return_value = async_instance
            yield {"sync": sync_mock, "async": async_mock}

    def test_list_models_returns_normalized(self, mock_http):
        client = MagicLLM(engine="openai", private_key="sk-test", model="gpt-4o")
        result = client.list_models()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "openai"

    def test_list_models_for_unsupported_engine_raises(self):
        """Unsupported engines raise NotImplementedError via resolver."""
        from magic_llm.base import MagicLlmBase
        with pytest.raises(NotImplementedError) as exc:
            MagicLlmBase._resolve_discovery_adapter(
                engine="definitely-not-supported",
                api_key="test",
            )
        assert "definitely-not-supported" in str(exc.value)
        assert "no discovery adapter" in str(exc.value)

    @pytest.mark.asyncio
    async def test_async_list_models(self, mock_http):
        client = MagicLLM(engine="openai", private_key="sk-test", model="gpt-4o")
        result = await client.async_list_models()
        assert len(result) == 2
        for m in result:
            assert m.external_id
            assert m.provider == "openai"

    @pytest.mark.asyncio
    async def test_sync_and_async_identical(self, mock_http):
        """Sync and async produce identical normalized output for same payload."""
        client = MagicLLM(engine="openai", private_key="sk-test", model="gpt-4o")
        sync_result = client.list_models()
        async_result = await client.async_list_models()
        assert len(sync_result) == len(async_result)
        for s, a in zip(sync_result, async_result):
            assert s.external_id == a.external_id
            assert s.provider == a.provider
            assert s.capabilities.chat == a.capabilities.chat


# ── Registry Completeness ────────────────────────────────────────────────

class TestOpenAIRegisteredOnImport:
    """Regression guard: ``openai`` adapter MUST be registered when the discovery
    package is imported — regardless of test execution order.

    The ``openai`` adapter registration lives in ``openai_discovery.py``, which
    is NOT imported by any adapter submodule — it MUST be explicitly imported
    by ``magic_llm/engine/discovery/__init__.py`` or it will never fire.

    This test verifies that fix independently of any other test file's imports.
    """

    def test_openai_registered_after_clean_import(self):
        from magic_llm.engine.discovery import get_adapter as ga
        cls = ga("openai")
        assert cls is not None, (
            "get_adapter('openai') returned None — "
            "openai_discovery is not imported by discovery/__init__.py"
        )

    def test_list_supported_engines_includes_openai(self):
        from magic_llm.engine.discovery import list_supported_engines as lse
        engines = lse()
        assert "openai" in engines, (
            "'openai' missing from list_supported_engines() — "
            "adapter was never registered"
        )

    def test_supports_discovery_true_for_openai(self):
        from magic_llm.engine.discovery import supports_discovery as sd
        assert sd("openai") is True


class TestRegistryCompleteness:
    """Every engine name from the prior registration block still resolves."""

    EXPECTED_ENGINES = [
        # OpenAI-compatible (own adapter per provider)
        "openai",
        "deepinfra",
        "groq",
        "novita",
        "perplexity",
        "together",
        "mistral",
        "deepseek",
        "hyperbolic",
        "cerebras",
        "xai",
        "parasail",
        "nebius",
        # Custom-endpoint providers
        "anthropic",
        "cohere",
        "google",
        "sambanova",
        "openrouter",
        # Azure surfaces
        "azure",
        "azure-foundry",
    ]

    def test_all_expected_engines_resolve(self):
        for name in self.EXPECTED_ENGINES:
            cls = get_adapter(name)
            assert cls is not None, (
                f"get_adapter('{name}') returned None — "
                "adapter not registered"
            )
            assert issubclass(cls, BaseDiscoveryAdapter), (
                f"get_adapter('{name}') returned {cls}, "
                "not a BaseDiscoveryAdapter subclass"
            )

    def test_each_adapter_is_concrete_subclass(self):
        """Each engine has its own adapter class (not the same class reused)."""
        classes = {}
        for name in self.EXPECTED_ENGINES:
            cls = get_adapter(name)
            classes[name] = cls
        # At minimum, these should be different classes:
        assert classes["openai"] is not classes["deepinfra"]
        assert classes["openai"] is not classes["anthropic"]
        assert classes["deepinfra"] is not classes["groq"]
        assert classes["groq"] is not classes["novita"]
        # Azure adapters must be separate classes (no shared "surface" heuristic)
        assert classes["azure"] is not classes["azure-foundry"]

    def test_list_supported_engines(self):
        engines = list_supported_engines()
        assert isinstance(engines, list)
        for name in self.EXPECTED_ENGINES:
            assert name in engines, (
                f"list_supported_engines() missing '{name}'"
            )


# ── Negative / Edge Cases ────────────────────────────────────────────────

class TestNegativeCases:
    """Error paths and edge cases."""

    def test_azure_speech_returns_none(self):
        """azure-speech was deregistered — get_adapter returns None."""
        cls = get_adapter("azure-speech")
        assert cls is None

    def test_unknown_engine_returns_none(self):
        cls = get_adapter("definitely-not-real")
        assert cls is None

    def test_supports_discovery_true_for_registered(self):
        from magic_llm.engine.discovery import supports_discovery
        assert supports_discovery("openai") is True
        assert supports_discovery("deepinfra") is True

    def test_supports_discovery_false_for_unregistered(self):
        from magic_llm.engine.discovery import supports_discovery
        assert supports_discovery("definitely-not-real") is False
        assert supports_discovery("azure-speech") is False


# ── Backward Compatibility ───────────────────────────────────────────────

class TestBackwardCompat:
    """Verify the public import path consumers (api.magic_llm) use."""

    def test_get_adapter_import_path(self):
        """Import exactly as api.magic_llm's discovery_client.py does."""
        from magic_llm.engine.discovery import get_adapter as ga
        assert ga is get_adapter  # same symbol

    def test_get_adapter_openai_returns_class(self):
        cls = get_adapter("openai")
        assert cls is not None
        assert issubclass(cls, BaseDiscoveryAdapter)

    def test_get_adapter_instantiation(self):
        """Verify adapter can be instantiated the way consumers do it."""
        cls = get_adapter("openai")
        adapter = cls(api_key="sk-test")
        assert adapter.provider == "openai"
        assert adapter._get_endpoint_url() == "https://api.openai.com/v1/models"


# ── Error Handling ───────────────────────────────────────────────────────

class TestErrorHandling:
    """HTTP error mapping to typed discovery exceptions."""

    @pytest.fixture
    def adapter(self):
        cls = get_adapter("openai")
        return cls(api_key="sk-test")

    def _mock_client_that_raises(self, error: HttpError):
        """Patch HttpClient.request to raise the given HttpError."""
        patcher = patch("magic_llm.engine.discovery.base_discovery.HttpClient")
        mock_cls = patcher.start()
        instance = MagicMock()
        instance.request.side_effect = error
        instance.__enter__.return_value = instance
        mock_cls.return_value = instance
        return patcher

    def test_401_raises_auth_error(self, adapter):
        patcher = self._mock_client_that_raises(
            HttpError("Unauthorized", status_code=401, response_content=b'{"error":"unauthorized"}')
        )
        try:
            with pytest.raises(DiscoveryAuthError) as exc:
                adapter.discover()
            assert exc.value.provider == "openai"
            assert exc.value.status_code == 401
        finally:
            patcher.stop()

    def test_429_raises_rate_limit_error(self, adapter):
        patcher = self._mock_client_that_raises(
            HttpError("Rate limited", status_code=429, response_content=b'{"error":"rate_limit"}')
        )
        try:
            with pytest.raises(DiscoveryRateLimitError) as exc:
                adapter.discover()
            assert exc.value.provider == "openai"
            assert exc.value.status_code == 429
        finally:
            patcher.stop()

    def test_429_with_retry_after(self, adapter):
        patcher = self._mock_client_that_raises(
            HttpError(
                "Rate limited",
                status_code=429,
                response_content=b'{"retry_after": 30}',
            )
        )
        try:
            with pytest.raises(DiscoveryRateLimitError) as exc:
                adapter.discover()
            assert exc.value.retry_after == 30
        finally:
            patcher.stop()

    def test_404_returns_empty_list(self, adapter):
        """Error Policy B: 404 returns empty list for all adapters."""
        patcher = self._mock_client_that_raises(
            HttpError("Not found", status_code=404, response_content=b'{"error":"not_found"}')
        )
        try:
            result = adapter.discover()
            assert result == []  # Graceful: 404 → []
        finally:
            patcher.stop()

    def test_500_raises_discovery_error(self, adapter):
        patcher = self._mock_client_that_raises(
            HttpError("Server error", status_code=500, response_content=b'internal error')
        )
        try:
            with pytest.raises(DiscoveryError) as exc:
                adapter.discover()
            assert exc.value.provider == "openai"
            assert exc.value.status_code == 500
        finally:
            patcher.stop()

    def test_invalid_json_raises_discovery_error(self, adapter):
        """Invalid JSON from provider → DiscoveryError."""
        patcher = patch("magic_llm.engine.discovery.base_discovery.HttpClient")
        mock_cls = patcher.start()
        instance = MagicMock()
        instance.request.return_value = b"not-json-at-all"
        instance.__enter__.return_value = instance
        mock_cls.return_value = instance
        try:
            with pytest.raises(DiscoveryError) as exc:
                adapter.discover()
            assert "invalid JSON" in str(exc.value).lower() or "invalid" in str(exc.value).lower()
        finally:
            patcher.stop()

    @pytest.mark.asyncio
    async def test_async_401_raises_auth_error(self, adapter):
        """Async path also maps errors correctly."""
        patcher = patch("magic_llm.engine.discovery.base_discovery.AsyncHttpClient")
        mock_cls = patcher.start()
        instance = MagicMock()
        instance.request = AsyncMock(
            side_effect=HttpError("Unauthorized", status_code=401, response_content=b'{}')
        )
        instance.__aenter__.return_value = instance
        mock_cls.return_value = instance
        try:
            with pytest.raises(DiscoveryAuthError):
                await adapter.async_discover()
        finally:
            patcher.stop()


# ── Credential Resolution ────────────────────────────────────────────────

class TestResolveCredentials:
    """``BaseDiscoveryAdapter.resolve_credentials()`` unit tests.

    Tests both the classmethod semantics and the integration through
    ``BaseChat.list_models()``.
    """

    @pytest.fixture
    def mock_http(self):
        with patch(
            "magic_llm.engine.discovery.base_discovery.HttpClient"
        ) as sync_mock:
            instance = MagicMock()
            instance.request.return_value = json.dumps(OPENAI_PAYLOAD).encode("utf-8")
            instance.__enter__.return_value = instance
            sync_mock.return_value = instance
            yield sync_mock

    def test_resolve_api_key_from_direct_attr(self):
        """api_key from chat_instance.api_key."""
        chat = MagicMock()
        chat.api_key = "sk-direct"
        chat.base = MagicMock()
        chat.base.api_key = "sk-base"
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["api_key"] == "sk-direct"

    def test_resolve_api_key_falls_back_to_base(self):
        """api_key falls back to chat_instance.base.api_key."""
        chat = MagicMock()
        chat.api_key = None
        chat.base = MagicMock()
        chat.base.api_key = "sk-base"
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["api_key"] == "sk-base"

    def test_resolve_empty_string_normalized_to_none(self):
        """Empty string api_key is normalized to None."""
        chat = MagicMock()
        chat.api_key = ""
        chat.base = MagicMock()
        chat.base.api_key = None
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["api_key"] is None

    def test_resolve_both_none(self):
        """Both api_key sources None → result is None."""
        chat = MagicMock()
        chat.api_key = None
        chat.base = MagicMock()
        chat.base.api_key = None
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["api_key"] is None

    def test_resolve_no_base_attr(self):
        """No 'base' attribute does not crash."""
        chat = MagicMock(spec=[])
        chat.api_key = "sk-direct"
        del chat.base
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["api_key"] == "sk-direct"

    def test_resolve_base_url_passthrough(self):
        """base_url is passed through as-is."""
        chat = MagicMock()
        chat.base_url = "https://custom.example.com/v1"
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["base_url"] == "https://custom.example.com/v1"

    def test_resolve_base_url_none(self):
        """base_url from chat is None by default."""
        chat = MagicMock()
        chat.base_url = None
        creds = BaseDiscoveryAdapter.resolve_credentials(chat)
        assert creds["base_url"] is None

    def test_list_models_integrates_resolve_credentials(self, mock_http):
        """End-to-end: BaseChat.list_models() delegates through resolve_credentials.

        This test verifies that ``BaseChat.list_models()`` calls
        ``BaseDiscoveryAdapter.resolve_credentials()`` and uses the
        returned api_key to build the adapter.
        """
        client = MagicLLM(engine="openai", private_key="sk-test", model="gpt-4o")
        result = client.list_models()
        assert len(result) == 2
        for m in result:
            assert m.provider == "openai"


# ── Base adapter bare-array guard ──────────────────────────────────────────

class TestBareArrayGuard:
    """Regression: _extract_raw_models must not crash on bare JSON arrays.

    Together, SambaNova, and Cohere return bare JSON arrays from their model endpoints.
    If a misconfigured provider record points at a non-overriding adapter, these must not
    raise ``AttributeError`` (the default ``.get("data", [])`` call fails on ``list``).

    Uses ``OpenAIDiscoveryAdapter`` (concrete, inherits base ``_extract_raw_models``).
    """

    def test_bare_array_returns_list_as_is(self):
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test", base_url="https://api.openai.com/v1/models")
        raw = [
            {"id": "model-a", "object": "model"},
            {"id": "model-b", "object": "model"},
        ]
        result = adapter._extract_raw_models(raw)
        assert result == raw
        assert len(result) == 2

    def test_bare_empty_array_returns_empty_list(self):
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test", base_url="https://api.openai.com/v1/models")
        result = adapter._extract_raw_models([])
        assert result == []

    def test_dict_data_still_works(self):
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test", base_url="https://api.openai.com/v1/models")
        raw = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        result = adapter._extract_raw_models(raw)
        assert len(result) == 2

    def test_dict_models_key_still_works(self):
        adapter = OpenAIDiscoveryAdapter(api_key="sk-test", base_url="https://api.openai.com/v1/models")
        raw = {"models": [{"id": "model-a"}]}
        result = adapter._extract_raw_models(raw)
        assert len(result) == 1
