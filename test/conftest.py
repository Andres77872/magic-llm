"""Shared pytest fixtures for the magic-llm test suite."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest


# ─── Key / Resource Fixtures ───────────────────────────────────────────────

KEYS_ENV_VAR = "MAGIC_LLM_KEYS"


def resolve_keys_file(*, required: bool = False) -> str | None:
    """Resolve an explicitly configured JSON credential file without reading it."""
    configured_path = os.getenv(KEYS_ENV_VAR)
    if not configured_path or not configured_path.strip():
        if not required:
            return None
        raise RuntimeError(f"Set {KEYS_ENV_VAR} to a private JSON credential file.")

    path = Path(configured_path).expanduser()
    if path.suffix.lower() != ".json":
        raise RuntimeError(f"{KEYS_ENV_VAR} must point to a .json file.")
    if not path.is_file():
        raise RuntimeError(f"{KEYS_ENV_VAR} does not point to an existing file.")
    return str(path)


@pytest.fixture(scope="session")
def keys_file_path() -> str:
    """Return the path to the keys file or skip selected live tests."""
    path = resolve_keys_file(required=False)
    if path is None:
        pytest.skip(f"Provider tests require {KEYS_ENV_VAR}=/path/to/keys.json.")
    return path


def load_keys_file(keys_file_path: str) -> Dict[str, Any]:
    """Load a provider-keyed JSON object without exposing credential values."""
    try:
        with open(keys_file_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError:
        raise RuntimeError("The configured credential file is not valid JSON.") from None
    if not isinstance(loaded, dict):
        raise RuntimeError("The configured credential JSON must contain a provider object.")
    return loaded


@pytest.fixture(scope="session")
def loaded_keys(keys_file_path: str) -> Dict[str, Any]:
    """Load keys lazily for explicitly selected live tests only."""
    return load_keys_file(keys_file_path)


@pytest.fixture(scope="session")
def provider_keys(loaded_keys: Dict[str, Any]) -> Dict[str, Any]:
    """Provider credentials for explicit live/provider tests.

    Credential values must never be printed. Tests should only mention provider
    names or key categories when skipping/failing selected live cases.
    """
    return loaded_keys


def get_provider_key(provider_keys: Dict[str, Any], provider: str, key_name: str) -> Dict[str, Any]:
    """Return one provider credential mapping or skip the selected live case."""
    entry = provider_keys.get(key_name)
    if entry is None:
        pytest.skip(f"Provider '{provider}' requires key category '{key_name}'.")
    if isinstance(entry, dict):
        return dict(entry)
    return {"private_key": entry}


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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
