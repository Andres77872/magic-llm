"""Public API regression tests for hard-removing legacy agentic names."""

import importlib

import pytest

from magic_llm import MagicLLM


def test_legacy_agentic_module_import_raises_module_not_found():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("magic_llm.util.agentic")


@pytest.mark.parametrize(
    "name",
    [
        "run_agentic",
        "run_agentic_stream",
        "run_agentic_async",
        "run_agentic_stream_async",
    ],
)
def test_legacy_run_agentic_util_exports_raise_import_error(name):
    with pytest.raises(ImportError):
        exec(f"from magic_llm.util import {name}", {})


@pytest.mark.parametrize("name", ["agentic", "agentic_stream"])
def test_legacy_magic_llm_client_attributes_raise_attribute_error(name):
    client = MagicLLM.__new__(MagicLLM)

    assert not hasattr(MagicLLM, name)
    assert not hasattr(client, name)
    with pytest.raises(AttributeError):
        getattr(client, name)
