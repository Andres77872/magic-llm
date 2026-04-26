import json
import os
import base64
from typing import Union

import pytest

from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from conftest import resolve_keys_file, DEFAULT_KEYS_FILE

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

# Providers with vision capabilities
VISION_PROVIDERS = [
    ("openai", "openai", {"model": "gpt-4o"}),
    ("google", "google", {"model": "gemini-2.0-flash"}),
    ("anthropic", "anthropic", {"model": "claude-3-7-sonnet-20250219"}),
]

# Resolve keys file with fallback — raises RuntimeError if missing
_KEYS_FILE = resolve_keys_file()
with open(_KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

# Sample image URL for testing
SAMPLE_IMAGE_URL = "https://img.arz.ai/4HZBWr2x.webp"

# Sample base64 encoded image — loaded from env var or skipped
_IMAGE_B64_FILE = os.getenv("MAGIC_LLM_IMAGE_B64_FILE")
if _IMAGE_B64_FILE and os.path.exists(_IMAGE_B64_FILE):
    SAMPLE_BASE64_IMAGE = open(_IMAGE_B64_FILE, 'r').read()
else:
    SAMPLE_BASE64_IMAGE = None

# Sample prompt to use with images
SAMPLE_PROMPT = "What do you see in this image?"

def get_sample_bytes_image():
    """Return a sample image as bytes"""
    if not SAMPLE_BASE64_IMAGE:
        pytest.skip("MAGIC_LLM_IMAGE_B64_FILE env var not set or file missing.")
    # Create a small image as bytes (decode the base64 sample)
    return base64.b64decode(SAMPLE_BASE64_IMAGE)

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_url(key_name, provider, kwargs):
    """Test adding an image from a URL"""
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image URL
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=SAMPLE_IMAGE_URL, media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Response for URL image: {resp.content[:100]}...")

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_base64(key_name, provider, kwargs):
    """Test adding an image as base64 string"""
    if not SAMPLE_BASE64_IMAGE:
        pytest.skip("MAGIC_LLM_IMAGE_B64_FILE env var not set or file missing.")
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with a base64 encoded image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=SAMPLE_BASE64_IMAGE, media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Response for base64 image: {resp.content[:100]}...")

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_bytes(key_name, provider, kwargs):
    """Test adding an image as bytes"""
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with a bytes image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=get_sample_bytes_image(), media_type="image/webp")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Response for bytes image: {resp.content[:100]}...")

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_multiple_images(key_name, provider, kwargs):
    """Test adding multiple images"""
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with multiple images of different types
    chat = ModelChat()
    chat.add_user_message(
        "Describe both of these images.",
        image=[SAMPLE_IMAGE_URL, get_sample_bytes_image()]
    )

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Response for multiple images: {resp.content[:100]}...")

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
def test_image_with_different_media_type(key_name, provider, kwargs):
    """Test adding an image with a different media type"""
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image and a different media type
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=get_sample_bytes_image(), media_type="image/png")

    # Generate a response
    resp = client.llm.generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Response for image with different media type: {resp.content[:100]}...")

def test_image_error_case():
    """Test error case when trying to add an image without content"""
    chat = ModelChat()

    # This should raise an exception
    with pytest.raises(Exception, match="Image cannot be alone"):
        chat.add_user_message("", image=SAMPLE_IMAGE_URL)

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    VISION_PROVIDERS,
    ids=[p[0] for p in VISION_PROVIDERS],
)
@pytest.mark.asyncio
async def test_async_image_generation(key_name, provider, kwargs):
    """Test async generation with an image"""
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)

    # Create a chat with an image
    chat = ModelChat()
    chat.add_user_message(SAMPLE_PROMPT, image=SAMPLE_IMAGE_URL)

    # Generate a response asynchronously
    resp = await client.llm.async_generate(chat)

    # Verify we got a response
    assert resp.content, "Expected non-empty content"
    print(f"Async response for image: {resp.content[:100]}...")