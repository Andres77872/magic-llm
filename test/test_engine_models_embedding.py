import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from magic_llm import MagicLLM

# Providers with embedding cap
EMBEDDING_PROVIDERS = [
    ("openai", "openai", {"model": "text-embedding-3-small"}),
    ("deepinfra", "openai", {"model": "BAAI/bge-m3", "encoding_format": "float"}),
    ("novita.ai", "openai", {"model": "baai/bge-m3", "encoding_format": "float"}),
    ("mistral", "openai", {"model": "mistral-embed"}),
    ("together.ai", "openai", {"model": "BAAI/bge-base-en-v1.5"}),
]

# Locate keys file via environment variable or default to test/keys.json
KEYS_FILE = os.getenv(
    "MAGIC_LLM_KEYS",
    "/home/andres/Documents/keys.json",
)
if not os.path.exists(KEYS_FILE):
    pytest.skip(
        f"No keys file found at {KEYS_FILE}. "
        "Set MAGIC_LLM_KEYS env var or place keys.json in this directory.",
        allow_module_level=True,
    )
with open(KEYS_FILE) as f:
    ALL_KEYS = json.load(f)

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
    print(resp)