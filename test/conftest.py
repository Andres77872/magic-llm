"""Shared pytest fixtures for the magic-llm test suite."""

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock, AsyncMock

import pytest


# ─── Key / Resource Fixtures ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def keys_file_path() -> str:
    """Return the path to the keys file, or empty string if not set."""
    return os.getenv("MAGIC_LLM_KEYS", "")


@pytest.fixture(scope="session")
def loaded_keys(keys_file_path: str) -> Dict[str, Any]:
    """Load keys from the keys file. Skips the entire module if file is missing."""
    if not keys_file_path or not os.path.exists(keys_file_path):
        pytest.skip(
            f"No keys file found at '{keys_file_path}'. "
            "Set MAGIC_LLM_KEYS env var to run integration tests.",
            allow_module_level=True,
        )
    with open(keys_file_path) as f:
        return json.load(f)


@pytest.fixture
def sample_audio_path() -> str:
    """Return path to sample audio file. Skips test if file is missing."""
    path = os.getenv("MAGIC_LLM_AUDIO_FILE", "")
    if not path or not os.path.exists(path):
        pytest.skip(
            f"No audio file found at '{path}'. "
            "Set MAGIC_LLM_AUDIO_FILE env var to run audio tests.",
        )
    return path


@pytest.fixture
def sample_image_b64() -> str:
    """Return base64-encoded image content. Skips test if file is missing."""
    path = os.getenv("MAGIC_LLM_IMAGE_B64_FILE", "")
    if not path or not os.path.exists(path):
        pytest.skip(
            f"No image b64 file found at '{path}'. "
            "Set MAGIC_LLM_IMAGE_B64_FILE env var to run image tests.",
        )
    with open(path, "r") as f:
        return f.read()


# ─── Mock HTTP Client Fixtures ─────────────────────────────────────────────

@pytest.fixture
def mock_sync_response():
    """Create a mock requests.Response object."""
    response = MagicMock()
    response.status_code = 200
    response.content = b'{"ok": true}'
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_sync_session(mock_sync_response):
    """Create a mock requests.Session that returns a configured response."""
    session = MagicMock()
    session.request.return_value.__enter__ = MagicMock(return_value=mock_sync_response)
    session.request.return_value.__exit__ = MagicMock(return_value=False)
    session.request.return_value = mock_sync_response
    return session


@pytest.fixture
def mock_async_response():
    """Create a mock aiohttp response context manager."""
    response = AsyncMock()
    response.status = 200
    response.read = AsyncMock(return_value=b'{"ok": true}')
    response.headers = {"Content-Type": "application/json"}
    # Make it work as an async context manager
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    return response


@pytest.fixture
def mock_async_session(mock_async_response):
    """Create a mock aiohttp.ClientSession that returns a configured response."""
    session = AsyncMock()
    session.request.return_value = mock_async_response
    return session


# ─── Model Chat Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def simple_chat():
    """Create a minimal ModelChat with one user message."""
    from magic_llm.model import ModelChat
    chat = ModelChat()
    chat.add_user_message("Hello, world!")
    return chat


@pytest.fixture
def simple_response():
    """Create a minimal valid ModelChatResponse."""
    from magic_llm.model.ModelChatResponse import (
        ModelChatResponse, Choice, Message, UsageModel,
    )
    return ModelChatResponse(
        id="test-123",
        object="chat.completion",
        created=1700000000.0,
        model="test-model",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=UsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def stream_chunk():
    """Create a minimal ChatCompletionModel stream chunk."""
    from magic_llm.model.ModelChatStream import (
        ChatCompletionModel, ChoiceModel, DeltaModel, UsageModel,
    )
    return ChatCompletionModel(
        id="chunk-1",
        model="test-model",
        choices=[
            ChoiceModel(
                index=0,
                delta=DeltaModel(content="Hello"),
                finish_reason=None,
            )
        ],
        usage=UsageModel(),
    )
