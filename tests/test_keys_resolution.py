"""Behavioral tests for explicit JSON credential-file resolution."""

import json

import pytest

from conftest import KEYS_ENV_VAR, load_keys_file, resolve_keys_file


def test_unconfigured_credentials_are_absent_until_live_tests_require_them(monkeypatch):
    monkeypatch.delenv(KEYS_ENV_VAR, raising=False)

    assert resolve_keys_file() is None
    with pytest.raises(RuntimeError, match=KEYS_ENV_VAR):
        resolve_keys_file(required=True)


def test_configured_json_path_uses_the_real_filesystem(monkeypatch, tmp_path):
    keys_path = tmp_path / "provider-keys.json"
    keys_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(KEYS_ENV_VAR, str(keys_path))

    assert resolve_keys_file() == str(keys_path)


@pytest.mark.parametrize("configured_name", ["missing.json", "keys.txt"])
def test_invalid_configured_path_fails_without_falling_back(monkeypatch, tmp_path, configured_name):
    configured_path = tmp_path / configured_name
    if configured_path.suffix != ".json":
        configured_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(KEYS_ENV_VAR, str(configured_path))

    with pytest.raises(RuntimeError, match=KEYS_ENV_VAR):
        resolve_keys_file()


def test_loaded_keys_accepts_only_a_json_provider_object(tmp_path):
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps({"openai": {"private_key": "sentinel"}}), encoding="utf-8")
    assert load_keys_file(str(valid_path)) == {
        "openai": {"private_key": "sentinel"},
    }

    for name, payload in (("invalid.json", "{"), ("list.json", "[]")):
        invalid_path = tmp_path / name
        invalid_path.write_text(payload, encoding="utf-8")
        with pytest.raises(RuntimeError, match="credential"):
            load_keys_file(str(invalid_path))
