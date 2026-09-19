import base64
import asyncio

import pytest

from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from conftest import get_provider_key

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Providers with vision capabilities
VISION_PROVIDERS = [
    ("openai", "openai", {"model": "gpt-4o"}),
    ("google", "google", {"model": "gemini-2.0-flash"}),
    ("anthropic", "anthropic", {"model": "claude-3-7-sonnet-20250219"}),
]

# Sample image URL for testing
SAMPLE_IMAGE_URL = "https://img.arz.ai/4HZBWr2x.webp"

# Sample prompt to use with images
SAMPLE_PROMPT = "What do you see in this image?"

def get_sample_bytes_image(sample_image_b64):
    """Return a sample image as bytes"""
    # Create a small image as bytes (decode the base64 sample)
    return base64.b64decode(sample_image_b64)


def _keys_for(provider_keys, provider, key_name):
    return get_provider_key(provider_keys, provider, key_name)

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_url(provider_keys, key_name, provider, kwargs):
    """Test adding an image from a URL"""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image URL
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=SAMPLE_IMAGE_URL, media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_base64(provider_keys, sample_image_b64, key_name, provider, kwargs):
    """Test adding an image as base64 string"""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with a base64 encoded image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=sample_image_b64, media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_bytes(provider_keys, sample_image_b64, key_name, provider, kwargs):
    """Test adding an image as bytes"""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with a bytes image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=get_sample_bytes_image(sample_image_b64), media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_multiple_images(provider_keys, sample_image_b64, key_name, provider, kwargs):
    """Test adding multiple images"""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with multiple images of different types
    chat = ModelChat()
    chat.add_user_message(
        "Describe both of these images.",
        image=[SAMPLE_IMAGE_URL, get_sample_bytes_image(sample_image_b64)]
    )

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_with_different_media_type(provider_keys, sample_image_b64, key_name, provider, kwargs):
    """Test adding an image with a different media type"""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image and a different media type
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=get_sample_bytes_image(sample_image_b64), media_type="image/png")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_async_vision_input_generation(provider_keys, key_name, provider, kwargs):
    """Test async chat generation with vision/image input (not image output generation)."""
    keys = _keys_for(provider_keys, provider, key_name)
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=SAMPLE_IMAGE_URL)

    # Generate a response asynchronously
    resp = asyncio.run(client.llm.async_generate(chat))

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
