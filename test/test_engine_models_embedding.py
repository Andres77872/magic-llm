import json
import os
import asyncio

import pytest

from magic_llm import MagicLLM

from conftest import resolve_keys_file, DEFAULT_KEYS_FILE

# Providers with embedding cap
EMBEDDING_PROVIDERS = [
    ("openai", "openai", {"model": "text-embedding-3-small"}),
    ("deepinfra", "openai", {"model": "BAAI/bge-m3", "encoding_format": "float"}),
    ("novita.ai", "openai", {"model": "baai/bge-m3", "encoding_format": "float"}),
    ("mistral", "openai", {"model": "mistral-embed"}),
    ("together.ai", "openai", {"model": "BAAI/bge-base-en-v1.5"}),
]

# Resolve keys file with fallback — raises RuntimeError if missing
_KEYS_FILE = resolve_keys_file()
with open(_KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

# All tests in this file require live provider access
pytestmark = pytest.mark.provider_functional

EXPECTED_TEXT = (
    "Dado que se trata de un único pago por el proyecto completo, debes tener en cuenta "
    "el valor a largo plazo que generará para el cliente, en lugar de solo el costo operativo o "
    "el tiempo invertido. Anteriormente, consideramos un escenario en que el cliente podía ahorrar "
    "entre 400 y 700 USD al mes en costos internos debido a la mayor precisión y eficiencia del sistema."
)

@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    EMBEDDING_PROVIDERS,
    ids=[p[0] for p in EMBEDDING_PROVIDERS],
)
def test_sync_embedding_single(key_name, provider, kwargs):
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)
    resp = client.llm.embedding(text=EXPECTED_TEXT)

    # resp is now a ModelEmbeddingResponse (not a raw dict)
    assert resp is not None
    assert len(resp.data) > 0, "Embedding response should contain at least one vector"
    assert len(resp.data[0].embedding) > 0, "Each embedding vector should be non-empty"
    assert all(isinstance(v, float) for v in resp.data[0].embedding), "Embedding values should be floats"


@pytest.mark.parametrize(
    ("key_name", "provider", "kwargs"),
    EMBEDDING_PROVIDERS,
    ids=[p[0] for p in EMBEDDING_PROVIDERS],
)
@pytest.mark.asyncio
async def test_async_embedding_single(key_name, provider, kwargs):
    keys = dict(ALL_KEYS[key_name])
    client = MagicLLM(**keys, **kwargs)
    resp = await client.llm.async_embedding(text=EXPECTED_TEXT)

    # resp is now a ModelEmbeddingResponse (not a raw dict)
    assert resp is not None
    assert len(resp.data) > 0, "Embedding response should contain at least one vector"
    assert len(resp.data[0].embedding) > 0, "Each embedding vector should be non-empty"
    assert all(isinstance(v, float) for v in resp.data[0].embedding), "Embedding values should be floats"
