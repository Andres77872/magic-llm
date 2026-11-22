# Azure

The `azure` engine is speech-only: it covers Azure AI Speech text-to-speech and speech-to-text.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="azure",
    speech_key="azure-speech-key",
    speech_region="eastus",
)
```

Both `speech_key` and `speech_region` are required; a missing value raises `ValueError` at construction. `private_key` is accepted as a `speech_key` alias only when `speech_key` is absent.

## Critical limitation

Chat is not implemented for `engine="azure"`. These methods all fail:

- `client.llm.generate(chat)`
- `client.llm.stream_generate(chat)`
- `client.llm.async_generate(chat)`
- `client.llm.async_stream_generate(chat)`

The engine raises `NotImplementedError` and the call is not retried. A configured `fallback` client handles the request; otherwise callers receive `ChatException` with `error_code="STREAM_GENERATION_ERROR"`, with the `NotImplementedError` as its `__context__`. Passing `tools` raises `TOOL_WIRING_NOT_IMPLEMENTED` instead.

Use the Azure engine only for speech APIs. Embeddings are not implemented either: `embedding()` returns `None`.

Azure Speech media methods are async-only. `audio_speech()` and `sync_audio_transcriptions()` raise clear unsupported-operation errors with async method hints.

## Text-to-speech example

```python
from magic_llm.model.ModelAudio import AudioSpeechRequest

request = AudioSpeechRequest(
    input="Hello from Magic LLM.",
    model="azure-speech",
    voice="en-US-JennyNeural",
    response_format="mp3",
)

audio_bytes = await client.llm.async_audio_speech(request)
```

Supported `response_format` values are `mp3`, `wav`, `pcm`, and `ogg`; they are mapped to Azure `X-Microsoft-OutputFormat` headers. Text is escaped before insertion into SSML.

## Speech-to-text

Azure exposes async transcription support through `async_audio_transcriptions`. Provide `language`, `filename`, and `content_type`; WAV (`audio/wav`) is required and non-WAV input is rejected before upload.

## Model discovery

`client.list_models()` does not work for `engine="azure"`. The `"azure"` discovery adapter lists **Azure OpenAI** models, not Speech resources, and it needs a `resource_name` the speech engine does not have. The call raises `ValueError("AzureOpenAIDiscoveryAdapter requires resource_name or base_url")`.

To list Azure model catalogs, use the discovery adapters directly:

```python
from magic_llm.engine.discovery import get_adapter

# Azure OpenAI data plane: GET https://{resource}.openai.azure.com/openai/models
azure_openai = get_adapter("azure")(resource_name="my-resource", api_key="azure-openai-key")
models = azure_openai.discover()

# Azure AI Foundry management plane, with an Entra ID bearer token
foundry = get_adapter("azure-foundry")(
    subscription_id="subscription-id",
    location="eastus",
    entra_token="entra-access-token",
)
models = foundry.discover()
```

Chat against Azure OpenAI deployments is not provided by the `azure` engine.
