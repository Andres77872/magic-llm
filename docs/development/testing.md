# Testing

Pytest discovers the entire suite in `tests/`. The default selection excludes `provider_functional` and `provider_health`. Pull requests and releases run this selection with branch coverage and a cold tokenizer cache; a failing test or coverage below the configured threshold blocks the release.

## Managed async stack

Install `pip install -e '.[test]'`. `setup.cfg` declares the supported ranges for pytest, pytest-asyncio, and pytest-cov. `pyproject.toml` enables automatic async tests with a function-scoped event loop. Budget tests use a deterministic byte estimator; tokenizer integration uses a small, fixed BPE vocabulary without downloading data.

## Marker taxonomy

| Marker | Meaning | Default offline? |
| --- | --- | --- |
| `provider_functional` | Explicit live/billable provider tests requiring credentials | Excluded |
| `provider_health` | Explicit live discovery/health smoke; also provider behavior | Excluded |
| `integration` | Offline multi-module seam tests with provider/network boundaries mocked | Included |

## Command matrix

| Purpose | Command | Live/billable? |
| --- | --- | --- |
| Default offline proof | `python -m pytest` | No |
| Collection safety | `python -m pytest --collect-only -q` | No |
| Async reliability slice | `python -m pytest tests/test_long_running_tool_budget.py` | No |
| Core media/protocol proof | `python -m pytest tests/test_media_fail_fast.py tests/test_stt_multipart.py tests/test_media_retry.py` | No |
| Chat retry and fallback proof | `python -m pytest tests/test_base_chat_decorators.py tests/test_unretryable_local_rejections.py tests/test_provider_attempt_control.py` | No |
| Provider media request shapes | `python -m pytest tests/test_openai_provider_media_shapes.py tests/test_provider_support_flags.py tests/test_google_tts_payload_shape.py` | No |
| Mock-quality / seam proof | `python -m pytest tests/test_amazon_engine_mocked.py tests/test_fireworks_hostnames.py tests/test_openrouter_pure_process_chunk.py tests/test_amazon_no_sdk.py tests/test_agent_loop_integration_smoke.py` | No |
| Discovery consolidation proof | `python -m pytest tests/test_discovery_integration.py` | No |
| Provider health smoke | `MAGIC_LLM_KEYS=/path/to/keys.json python -m pytest tests/test_discovery_smoke.py -m provider_health` | Yes — explicit live |
| Provider-functional smoke | `MAGIC_LLM_KEYS=/path/to/keys.json python -m pytest -m "provider_functional and not provider_health"` | Yes — explicit live |
| Coverage gate | `python -m pytest --cov=magic_llm --cov-branch` | No; enforced at the configured threshold |

## Credentials and resource fixtures

Live tests load credentials lazily through fixtures only after explicit live marker selection. `MAGIC_LLM_KEYS` is required and must point to an existing `.json` file; no local fallback is consulted.

Missing credentials skip selected live tests with provider/key-category messages. Offline collection must not open or parse credential files.

Optional live media fixtures:

| Variable | Purpose |
| --- | --- |
| `MAGIC_LLM_KEYS` | Path to JSON file with provider keys for explicit live jobs. |
| `MAGIC_LLM_AUDIO_FILE` | Local audio fixture path for explicit provider-functional audio tests. |
| `MAGIC_LLM_IMAGE_B64_FILE` | Local file containing base64 image data for explicit provider-functional vision tests. |

Never print, copy, commit, or persist credential contents in logs, test output, docs, or SDD/RDD artifacts.

## Cleanup guardrails

- Delete-without-replacement is forbidden.
- Every cleanup target must be classified as `keep`, `rewrite`, `consolidate`, `quarantine`, `delete-after-replacement`, or `not-delete`.
- Deletion requires `deleted test -> replacement test -> offline verification command`.
- Protected proof files include media fail-fast, STT multipart, media retry, Azure speech, discovery capabilities, Amazon Polly, public API removal, and discovery facade tests.
- Live image/audio/embedding/tokenizer smoke tests are exploratory/provider-functional smoke; they do not replace offline protocol/request-shape proof.

## Debug environment variables

| Variable | Purpose |
| --- | --- |
| `MAGIC_LLM_DEBUG_PAYLOAD` | Log compact payload summaries. |
| `MAGIC_LLM_DEBUG_PAYLOAD_FULL` | Log full payload JSON. |

Full payload logging can leak prompts or data. Use it only in safe local debugging sessions, never in CI/live artifacts with secrets.
