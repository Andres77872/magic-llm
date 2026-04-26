"""Tests for Fireworks ASR hostname updates."""
import inspect

import pytest

from magic_llm.engine.openai_adapters.openai_fireworks import ProviderFireworks


class TestFireworksHostnames:
    """Test that Fireworks uses current *.api.fireworks.ai hostnames for ASR."""

    def test_whisper_v3_uses_current_hostname_in_source(self):
        """whisper-v3 ASR URL uses audio-prod.api.fireworks.ai in source code."""
        source = inspect.getsource(ProviderFireworks.async_audio_transcriptions)
        assert "https://audio-prod.api.fireworks.ai/v1" in source
        assert "us-virginia-1.direct.fireworks.ai" not in source

    def test_whisper_v3_turbo_uses_current_hostname_in_source(self):
        """whisper-v3-turbo ASR URL uses audio-turbo.api.fireworks.ai in source code."""
        source = inspect.getsource(ProviderFireworks.async_audio_transcriptions)
        assert "https://audio-turbo.api.fireworks.ai/v1" in source
        assert "us-virginia-1.direct.fireworks.ai" not in source

    def test_no_deprecated_hostname_in_module(self):
        """No deprecated hostname appears in the module source."""
        import magic_llm.engine.openai_adapters.openai_fireworks as mod
        source = inspect.getsource(mod)
        assert "us-virginia-1.direct.fireworks.ai" not in source

    def test_base_url_is_general_inference_endpoint(self):
        """Provider base_url is the general inference endpoint (not ASR-specific)."""
        provider = ProviderFireworks(api_key="test")
        assert provider.base_url == "https://api.fireworks.ai/inference/v1"
