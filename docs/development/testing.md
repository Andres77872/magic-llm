# Testing

Project test command:

```bash
pytest test/ -v
```

Do not run integration tests without explicit provider keys and cost awareness.

## Pytest configuration

`pyproject.toml` configures:

- `asyncio_mode = "auto"`
- `testpaths = ["test"]`
- coverage source: `magic_llm`
- coverage fail-under: `60`

Markers:

- `provider_health`
- `provider_functional`
- `timeout`

## Environment variables

| Variable | Purpose |
| --- | --- |
| `MAGIC_LLM_KEYS` | Path to JSON file with provider keys. |
| `MAGIC_LLM_AUDIO_FILE` | Audio fixture path for audio tests. |
| `MAGIC_LLM_IMAGE_B64_FILE` | Base64 image fixture path for vision tests. |

## Debug environment variables

| Variable | Purpose |
| --- | --- |
| `MAGIC_LLM_DEBUG_PAYLOAD` | Log compact payload summaries. |
| `MAGIC_LLM_DEBUG_PAYLOAD_FULL` | Log full payload JSON. |

Full payload logging can leak prompts or data. Use it only in safe local debugging sessions.
