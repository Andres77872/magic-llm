# Azure

The `azure` engine is speech-only in Magic LLM v0.1.37.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="azure",
    speech_key="azure-speech-key",
    speech_region="eastus",
)
```

## Critical limitation

These chat methods raise `NotImplementedError` for `engine="azure"`:

- `client.llm.generate(chat)`
- `client.llm.stream_generate(chat)`
- `client.llm.async_generate(chat)`
- `client.llm.async_stream_generate(chat)`

Use the Azure engine only for speech APIs.

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
