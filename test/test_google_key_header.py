"""Tests for Google API key header auth migration."""
import pytest

from magic_llm.engine.engine_google import EngineGoogle
from magic_llm.model import ModelChat


class TestGoogleKeyNotInUrls:
    """Test that EngineGoogle URLs do not contain API key."""

    def test_url_no_key(self):
        """self.url does not contain key= query parameter."""
        engine = EngineGoogle(api_key="test-key-123", model="gemini-pro")
        assert "key=" not in engine.url
        assert engine.url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def test_url_stream_no_key(self):
        """self.url_stream does not contain key= query parameter."""
        engine = EngineGoogle(api_key="test-key-123", model="gemini-pro")
        assert "key=" not in engine.url_stream
        assert engine.url_stream == "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?alt=sse"

    def test_url_tts_no_key(self):
        """self.url_tts does not contain key= query parameter."""
        engine = EngineGoogle(api_key="test-key-123", model="gemini-pro")
        assert "key=" not in engine.url_tts
        assert engine.url_tts == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"


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

    @pytest.mark.asyncio
    async def test_prepare_data_async_has_header(self):
        """prepare_data() headers include x-goog-api-key."""
        engine = EngineGoogle(api_key="my-secret-key", model="gemini-pro")
        chat = ModelChat()
        chat.add_message("user", "hello")
        _, headers, _ = await engine.prepare_data(chat)
        assert "x-goog-api-key" in headers
        assert headers["x-goog-api-key"] == "my-secret-key"

    def test_header_value_matches_api_key(self):
        """x-goog-api-key header value matches the provided API key."""
        engine = EngineGoogle(api_key="special-key-abc123", model="gemini-pro")
        chat = ModelChat()
        chat.add_message("user", "test")
        _, headers, _ = engine.prepare_data_sync(chat)
        assert headers["x-goog-api-key"] == "special-key-abc123"
