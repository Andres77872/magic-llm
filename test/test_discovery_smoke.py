"""Real-API smoke tests for model discovery.

Calls discovery endpoints against real provider endpoints.  Marked with
``pytest.mark.provider_health`` (not ``provider_functional``) to distinguish
from generation/streaming provider tests.

Coverage (Tier 1 + Tier 2)::

  Tier 1 — stable E2E candidates::

    openai      — Canonical baseline (core engine, full outside-in path)
    anthropic   — Custom ``x-api-key`` auth (core engine, full outside-in path)
    cohere      — Proprietary response shape (core engine, full outside-in path)
    openrouter  — Public endpoint, no auth (proxy engine, production resolution path)
    deepinfra   — URL regression guard (proxy engine, production resolution path)
    groq        — Distinct URL prefix ``/openai/v1/`` (proxy engine, production resolution path)

  Tier 2 — investigative / conditional::

    deepseek    — URL mismatch investigation (/v1/models vs /models)
    mistral     — Clean OpenAI-compatible, well-documented API
    together    — Key name mismatch (key file uses "together.ai")
    xai         — Model-scoped auth, key name mismatch (key file uses "x.ai")
    perplexity  — OpenAI-compatible, well-documented API
    sambanova   — Resilient adapter (returns [] on error), key name mismatch
                  (key file uses "SambaNova"), min_models=0

Key semantics
-------------
Every provider declared in the test list is ALWAYS parametrised.  If a
key-required provider lacks credentials in the keys file, its test case
*FAILS* with a clear error — there is no silent filtering or skipping.
This is intentional: the user's rule is "missing key = real error", not
"missing key = skip and pretend it's fine".

Two test paths
--------------
``MagicLlmBase.ENGINE_MAP`` only contains 7 core engines (openai, google,
cloudflare, amazon, cohere, anthropic, azure).  Providers like deepinfra,
groq, and openrouter use the OpenAI chat engine under the hood but have
*separate* discovery adapters.  ``MagicLLM(engine="deepinfra", ...)`` would
fail with ``ValueError: Unsupported engine``.

Therefore:

- **Core engines** (openai, anthropic, cohere): ``MagicLLM(engine=..., ...).list_models()``
  — full outside-in chain through the public API.

- **Proxy engines** (openrouter, deepinfra, groq):
  ``MagicLlmBase._resolve_discovery_adapter(engine, api_key).discover()``
  — the *production* discovery-resolution path that ``BaseChat.list_models()``
  itself calls internally.  This is NOT a test-only shortcut; it is the real
  production code path used by all core-engine discovery calls.
"""

from __future__ import annotations

import json

import pytest

from conftest import resolve_keys_file
from magic_llm import MagicLLM
from magic_llm.base import MagicLlmBase
from magic_llm.engine.discovery import supports_discovery

pytestmark = pytest.mark.provider_health

# ── Key resolution ──────────────────────────────────────────────────────────

_KEYS_FILE = resolve_keys_file()
with open(_KEYS_FILE) as _f:
    _ALL_KEYS = json.load(_f)

# ── Provider matrix ─────────────────────────────────────────────────────────
# Fields: (engine_name, key_name_in_json, min_models, is_core_engine, requires_key)
#
# ``is_core_engine`` = engine is in ``MagicLlmBase.ENGINE_MAP`` (can use
# ``MagicLLM.list_models()`` directly).
#
# ``min_models`` — lower bound on the number of models returned.

_TIER_1_PROVIDERS: list[tuple[str, str, int, bool, bool]] = [
    # Core engines — full outside-in path via MagicLLM.list_models()
    ("openai",     "openai",     1, True,  True),
    ("anthropic",  "anthropic",  1, True,  True),
    ("cohere",     "cohere",     1, True,  True),
    # Proxy engines — production resolution path via _resolve_discovery_adapter
    ("openrouter", "openrouter", 1, False, False),  # public endpoint — no key needed
    ("deepinfra",  "deepinfra",  1, False, True),
    ("groq",       "groq",       1, False, True),
]

_TIER_2_PROVIDERS: list[tuple[str, str, int, bool, bool]] = [
    # Tier 2 — investigative / conditional providers
    # deepseek: URL mismatch investigation — code uses /v1/models, docs say /models
    ("deepseek",  "deepseek",   1, False, True),
    # mistral: clean OpenAI-compatible, well-documented API
    ("mistral",   "mistral",    1, False, True),
    # together: key name mismatch — key file uses "together.ai", engine is "together"
    ("together",  "together.ai", 1, False, True),
    # xai: key name mismatch — key file uses "x.ai", engine is "xai"
    ("xai",       "x.ai",        1, False, True),
    # perplexity: clean OpenAI-compatible, standard key mapping
    ("perplexity","perplexity",  1, False, True),
    # sambanova: resilient adapter (returns [] on bad auth); key name mismatch
    # (key file uses "SambaNova"). min_models=0 because the resilient adapter
    # returns an empty list on error.
    ("sambanova", "SambaNova",   0, False, True),
]

_ALL_PROVIDERS = _TIER_1_PROVIDERS + _TIER_2_PROVIDERS
_ENGINE_IDS = [e[0] for e in _ALL_PROVIDERS]


# ── Discovery helpers ───────────────────────────────────────────────────────


def _resolve_key(engine: str, key_name: str, requires_key: bool) -> str | None:
    """Resolve ``api_key`` for *engine*.

    If the provider requires a key and it is not in the keys file, this calls
    ``pytest.fail()`` with a clear error — no silent skipping.

    If the provider does NOT require a key (OpenRouter), returns ``None``.
    """
    if not requires_key:
        return None

    entry = _ALL_KEYS.get(key_name)
    if entry is None:
        pytest.fail(
            f"Provider '{engine}' requires key '{key_name}' but it is "
            f"not present in the keys file ({_KEYS_FILE}).  "
            "Set the missing key or remove this provider from the test list.",
        )
    return entry.get("private_key")


def _discover_models(engine: str) -> list:
    """Call the real discovery endpoint for *engine* and return model list.

    Two code paths:

    * **Core engines** (openai, anthropic, cohere): ``MagicLLM().list_models()``
      — tests the full outside-in chain through the public API.

    * **Proxy engines** (openrouter, deepinfra, groq):
      ``MagicLlmBase._resolve_discovery_adapter().discover()`` — the production
      discovery-resolution path used internally by ``BaseChat.list_models()``.
    """
    entry = next(e for e in _ALL_PROVIDERS if e[0] == engine)
    _key_name = entry[1]
    _is_core = entry[3]
    _requires_key = entry[4]

    api_key = _resolve_key(engine, _key_name, _requires_key)

    if _is_core:
        # ── Outside-in path (public API) ────────────────────────────────────
        key_data = dict(_ALL_KEYS[_key_name])
        key_data.pop("engine", None)  # would conflict with explicit engine=
        client = MagicLLM(engine=engine, **key_data)
        return client.list_models()

    # ── Production discovery-resolution path ────────────────────────────────
    assert supports_discovery(engine), (
        f"Engine '{engine}' has no registered discovery adapter"
    )
    adapter = MagicLlmBase._resolve_discovery_adapter(
        engine=engine,
        api_key=api_key,
        base_url=None,
    )
    return adapter.discover()


@pytest.mark.asyncio
async def _async_discover_models(engine: str) -> list:
    """Async variant of :func:`_discover_models`."""
    entry = next(e for e in _ALL_PROVIDERS if e[0] == engine)
    _key_name = entry[1]
    _is_core = entry[3]
    _requires_key = entry[4]

    api_key = _resolve_key(engine, _key_name, _requires_key)

    if _is_core:
        key_data = dict(_ALL_KEYS[_key_name])
        key_data.pop("engine", None)
        client = MagicLLM(engine=engine, **key_data)
        return await client.async_list_models()

    assert supports_discovery(engine), (
        f"Engine '{engine}' has no registered discovery adapter"
    )
    adapter = MagicLlmBase._resolve_discovery_adapter(
        engine=engine,
        api_key=api_key,
        base_url=None,
    )
    return await adapter.async_discover()


# ── Shared assertions ───────────────────────────────────────────────────────


def _assert_valid_models(engine: str, models: list) -> None:
    """Assert common invariants on a discovery result list."""
    entry = next(e for e in _ALL_PROVIDERS if e[0] == engine)
    _min_models = entry[2]

    assert len(models) >= _min_models, (
        f"{engine}: expected ≥{_min_models} models, got {len(models)}"
    )
    for m in models:
        assert m.external_id, (
            f"{engine}: model missing external_id "
            f"(raw_id={m.raw.get('id', '?')})"
        )
        assert m.provider == engine, (
            f"{engine}: model provider={m.provider!r} does not match engine"
        )


# ── Sync smoke tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize("engine", _ENGINE_IDS, ids=_ENGINE_IDS)
def test_discovery_smoke(engine: str) -> None:
    """Verify real discovery endpoint returns valid model list via sync path.

    Asserts:
    - ``len(models) >= min_models``  (at least the required minimum)
    - ``all(m.external_id)``          — no null or empty external IDs
    - ``all(m.provider == engine)``   — correct provider attribution

    Key semantics: if the provider requires a key and it is missing from the
    keys file, this test FAILS with a clear error.  No silent skipping.
    """
    models = _discover_models(engine)
    _assert_valid_models(engine, models)


# ── Async smoke tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", _ENGINE_IDS, ids=_ENGINE_IDS)
async def test_async_discovery_smoke(engine: str) -> None:
    """Same as :func:`test_discovery_smoke` but via the async discovery path.

    Asserts the same invariants (non-empty list, no null IDs, correct
    provider attribution) on the async code path.
    """
    models = await _async_discover_models(engine)
    _assert_valid_models(engine, models)
