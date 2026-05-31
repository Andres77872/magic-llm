# Audio

Magic LLM exposes speech request models in `magic_llm.model.ModelAudio`.

## Speech-to-text

```python
from magic_llm import MagicLLM
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest

client = MagicLLM(engine="openai", private_key="sk-your-key")

with open("speech.mp3", "rb") as f:
    request = AudioTranscriptionsRequest(
        file=f.read(),
        model="whisper-1",
        response_format="json",
    )

result = client.llm.sync_audio_transcriptions(request)
print(result)
```

Async:

```python
result = await client.llm.async_audio_transcriptions(request)
```

## Text-to-speech

```python
from magic_llm.model.ModelAudio import AudioSpeechRequest

request = AudioSpeechRequest(
    input="Hello from Magic LLM.",
    model="tts-1",
    voice="alloy",
    response_format="mp3",
    speed=1.0,
)

audio_bytes = client.llm.audio_speech(request)
```

Async:

```python
audio_bytes = await client.llm.async_audio_speech(request)
```

## Provider notes

- OpenAI-compatible audio support exists for selected adapters such as OpenAI, DeepInfra, and Fireworks.
- Azure is speech-only: chat methods are not implemented, but speech methods are the intended use.
- Not all providers support both transcription and speech synthesis.
