"""Tests for OpenRouter engine-level usage polling."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_llm.engine.engine_openai import EngineOpenAI
from magic_llm.engine.openai_adapters.openai_openrouter import ProviderOpenRouter
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class TestOpenRouterEnginePolling:
    """Test that OpenRouter usage polling happens at engine level."""

    @pytest.mark.asyncio
    async def test_async_stream_generate_polls_when_id_generation_set(self):
        """async_stream_generate polls /generation endpoint when id_generation is captured."""
        engine = EngineOpenAI(api_key="test", model="openai/gpt-4")
        # Replace the base provider with OpenRouter
        engine.base = ProviderOpenRouter(api_key="test")

        # Mock the streaming response to yield one chunk with an id
        chunk_data = {
            "id": "gen-abc123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "openai/gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }

        # Mock AsyncHttpClient.post_stream to yield SSE chunks
        async def mock_stream(*args, **kwargs):
            yield f'data: {json.dumps(chunk_data)}'.encode()
            yield b"data: [DONE]"

        # Mock AsyncHttpClient.request for the polling call
        poll_response = {
            "data": {
                "generationId": "gen-abc123",
                "model": "openai/gpt-4",
                "tokensPrompt": 10,
                "tokensCompletion": 5,
                "totalCost": "0.001",
            }
        }

        with patch("magic_llm.engine.engine_openai.AsyncHttpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post_stream = mock_stream
            mock_client.request = AsyncMock(return_value=json.dumps(poll_response).encode())
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            # Mock the provider's process_chunk
            with patch.object(engine.base, "process_chunk") as mock_process:
                mock_chunk = ChatCompletionModel(**chunk_data)
                mock_process.return_value = mock_chunk

                chunks = []
                async for chunk in engine.async_stream_generate(
                    ModelChat(system="test")
                ):
                    chunks.append(chunk)

                # Verify polling request was made
                assert mock_client.request.called
                call_args = mock_client.request.call_args
                assert "openrouter.ai/api/v1/generation" in call_args[0][1]
                assert "gen-abc123" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_polling_when_no_id_generation(self):
        """async_stream_generate does not poll when no id_generation captured."""
        engine = EngineOpenAI(api_key="test", model="openai/gpt-4")
        engine.base = ProviderOpenRouter(api_key="test")

        # Chunk without a meaningful id (empty string means no polling)
        chunk_data = {
            "id": "",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "openai/gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }

        async def mock_stream(*args, **kwargs):
            yield f'data: {json.dumps(chunk_data)}'.encode()
            yield b"data: [DONE]"

        with patch("magic_llm.engine.engine_openai.AsyncHttpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post_stream = mock_stream
            mock_client.request = AsyncMock(return_value=b"{}")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch.object(engine.base, "process_chunk") as mock_process:
                mock_chunk = ChatCompletionModel(**chunk_data)
                mock_process.return_value = mock_chunk

                chunks = []
                async for chunk in engine.async_stream_generate(
                    ModelChat(system="test")
                ):
                    chunks.append(chunk)

                # Verify NO polling request was made
                assert not mock_client.request.called

    @pytest.mark.asyncio
    async def test_polling_failure_does_not_crash(self):
        """Polling failure is logged and doesn't crash the stream."""
        engine = EngineOpenAI(api_key="test", model="openai/gpt-4")
        engine.base = ProviderOpenRouter(api_key="test")

        chunk_data = {
            "id": "gen-abc123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "openai/gpt-4",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }

        async def mock_stream(*args, **kwargs):
            yield f'data: {json.dumps(chunk_data)}'.encode()
            yield b"data: [DONE]"

        with patch("magic_llm.engine.engine_openai.AsyncHttpClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post_stream = mock_stream
            # Simulate polling failure
            mock_client.request = AsyncMock(side_effect=Exception("Network error"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with patch.object(engine.base, "process_chunk") as mock_process:
                mock_chunk = ChatCompletionModel(**chunk_data)
                mock_process.return_value = mock_chunk

                # Should not raise
                chunks = []
                async for chunk in engine.async_stream_generate(
                    ModelChat(system="test")
                ):
                    chunks.append(chunk)

                assert len(chunks) >= 1
