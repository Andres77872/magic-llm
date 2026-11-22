"""Opt-in live health checks through the same public client that applications use."""
import pytest

from conftest import get_provider_key
from magic_llm import MagicLLM

pytestmark = [pytest.mark.provider_health, pytest.mark.provider_functional]

# Discovery provider, credential category (None for public listings), chat base URL.
PROVIDERS = [
    ("openai", "openai", None),
    ("anthropic", "anthropic", None),
    ("cohere", "cohere", None),
    ("openrouter", None, "https://openrouter.ai/api/v1"),
    ("deepinfra", "deepinfra", "https://api.deepinfra.com/v1/openai"),
    ("groq", "groq", "https://api.groq.com/openai/v1"),
    ("deepseek", "deepseek", "https://api.deepseek.com/v1"),
    ("mistral", "mistral", "https://api.mistral.ai/v1"),
    ("together", "together.ai", "https://api.together.xyz/v1"),
    ("xai", "x.ai", "https://api.x.ai/v1"),
    ("perplexity", "perplexity", "https://api.perplexity.ai"),
    ("sambanova", "SambaNova", "https://api.sambanova.ai/v1"),
]


@pytest.mark.parametrize("provider,key_name,base_url", PROVIDERS, ids=[p[0] for p in PROVIDERS])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_discovery_smoke(request, provider, key_name, base_url, asynchronous):
    key = None
    if key_name is not None:
        keys = get_provider_key(request.getfixturevalue("provider_keys"), provider, key_name)
        key = keys["private_key"]
    options = {"base_url": base_url} if base_url else {}
    client = MagicLLM(engine="openai" if base_url else provider, private_key=key, **options)
    try:
        models = await client.async_list_models() if asynchronous else client.list_models()
        assert models, f"{provider}: expected a non-empty model listing"
        assert all(m.external_id and m.provider == provider for m in models)
    finally:
        client.llm.executor.shutdown(wait=False)
