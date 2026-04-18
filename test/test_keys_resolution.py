"""Characterization tests for resolve_keys_file() key resolution logic.

Proves all branches of the key-resolution helper:
1. Env set, file exists → use env path
2. Env set, file missing → use fallback
3. Env unset, fallback exists → use fallback
4. Env unset, fallback missing → raise RuntimeError
5. Env empty string, fallback exists → use fallback
"""

import os
from unittest.mock import patch

import pytest

# Import from conftest — the module where resolve_keys_file lives.
# conftest.py is auto-imported by pytest; we import it explicitly for testing.
from conftest import resolve_keys_file, DEFAULT_KEYS_FILE


class TestResolveKeysFile:
    """All tests mock os.getenv and os.path.exists to prove branch coverage.

    Patch targets are 'os.getenv' and 'os.path.exists' at the module level
    since conftest.py uses 'import os' and calls os.getenv/os.path.exists directly.
    """

    def test_env_set_file_exists_returns_env_path(self):
        """Scenario 1: MAGIC_LLM_KEYS is set and the file exists → use env path."""
        env_path = "/tmp/keys.json"
        with patch("os.getenv", return_value=env_path), \
             patch("os.path.exists", return_value=True):
            result = resolve_keys_file()
        assert result == env_path

    def test_env_set_file_missing_returns_fallback(self):
        """Scenario 2: MAGIC_LLM_KEYS is set but file doesn't exist → use fallback."""
        env_path = "/tmp/keys.json"

        def exists_side_effect(path):
            return path == DEFAULT_KEYS_FILE

        with patch("os.getenv", return_value=env_path), \
             patch("os.path.exists", side_effect=exists_side_effect):
            result = resolve_keys_file()
        assert result == DEFAULT_KEYS_FILE

    def test_env_unset_fallback_exists_returns_fallback(self):
        """Scenario 3: MAGIC_LLM_KEYS is unset but fallback exists → use fallback."""
        with patch("os.getenv", return_value=None), \
             patch("os.path.exists", return_value=True):
            result = resolve_keys_file()
        assert result == DEFAULT_KEYS_FILE

    def test_env_unset_fallback_missing_raises_runtime_error(self):
        """Scenario 4: MAGIC_LLM_KEYS is unset and fallback doesn't exist → RuntimeError."""
        with patch("os.getenv", return_value=None), \
              patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="No API keys file found"):
                resolve_keys_file()

    def test_env_empty_string_fallback_exists_returns_fallback(self):
        """Scenario 5: MAGIC_LLM_KEYS is empty string but fallback exists → use fallback."""
        with patch("os.getenv", return_value=""), \
             patch("os.path.exists", return_value=True):
            result = resolve_keys_file()
        assert result == DEFAULT_KEYS_FILE
