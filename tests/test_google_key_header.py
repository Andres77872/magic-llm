"""Tests for Google API key header auth migration."""
import asyncio

import pytest

from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.model import ModelChat


class TestGoogleKeyNotInUrls:
    """Test that EngineGoogle URLs do not contain API key."""

    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            (
                "url",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
            ),
            (
                "url_stream",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?alt=sse",
            ),
            (
                "url_tts",
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent",
            ),
        ],
    )
    def test_api_key_is_absent_from_urls(self, attribute, expected):
        engine = EngineGoogle(api_key="test-key-123", model="gemini-pro")
        url = getattr(engine, attribute)

        assert "key=" not in url
        assert url == expected


class TestGoogleKeyInHeaders:
    """Test that API key is passed via x-goog-api-key header."""

    def test_prepare_data_sync_has_header(self):
        """prepare_data_sync() headers include x-goog-api-key."""
        engine = EngineGoogle(api_key="my-secret-key", model="gemini-pro")
        chat = ModelChat()
        chat.add_message("user", "hello")
        _, headers, _ = engine.prepare_data_sync(chat)
        assert "x-goog-api-key" in headers
        assert headers["x-goog-api-key"] == "my-secret-key"

    def test_prepare_data_async_has_header(self):
        """prepare_data() headers include x-goog-api-key."""
        engine = EngineGoogle(api_key="my-secret-key", model="gemini-pro")
        chat = ModelChat()
        chat.add_message("user", "hello")
        _, headers, _ = asyncio.run(engine.prepare_data(chat))
        assert "x-goog-api-key" in headers
        assert headers["x-goog-api-key"] == "my-secret-key"
