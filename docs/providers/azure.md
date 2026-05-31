# Azure

The `azure` engine is speech-only in Magic LLM v0.1.36.

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

## Speech-to-text

Azure exposes async transcription support through `async_audio_transcriptions`. Check provider behavior in your region and model setup.
