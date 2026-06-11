# Audio

Magic LLM exposes speech request models in `magic_llm.model.ModelAudio`.

Unsupported media paths now fail fast with `UNSUPPORTED_MEDIA_OPERATION` instead of returning `None`. This is an intentional bug-fix breaking change: branch from the matrix below before calling a media method.

## Provider × method matrix

| Engine / provider path | TTS sync | TTS async | STT sync | STT async | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `openai` official OpenAI | ❌ | ✅ | ✅ | ✅ | Sync TTS raises; use `async_audio_speech`. |
| `openai` + Together | ✅ | ✅ | ❌ | ❌ | `/audio/generations` TTS only. |
| `openai` + DeepInfra | ❌ | ✅ | ❌ | ❌ | Async TTS only; STT is conservative/unsupported. |
| `openai` + Fireworks | ❌ | ❌ | ❌ | ✅ | Async STT only for `whisper-v3` and `whisper-v3-turbo`. |
| Other OpenAI-compatible adapters | ❌ | ❌ | ❌ | ❌ | No inherited generic media endpoint claim. |
| `google` | ✅ | ✅ | ❌ | ❌ | Gemini TTS returns WAV bytes. |
| `azure` Speech | ❌ | ✅ | ❌ | ✅ | Async-only Azure Speech. |
| `amazon` Polly | ✅ | ❌ | ❌ | ❌ | Sync TTS only; no Amazon Transcribe in core. |
| `anthropic`, `cloudflare`, `cohere` | ❌ | ❌ | ❌ | ❌ | Audio unsupported. |

## Speech-to-text

STT uploads carry audio bytes **plus metadata**. Provide accurate `filename` and `content_type` when possible; the library can infer common WAV/MP3/OGG/FLAC/M4A signatures but will reject unknown bytes rather than lying with fake MP3/WAV metadata.

```python
from magic_llm import MagicLLM
from magic_llm.model.ModelAudio import AudioTranscriptionsRequest

client = MagicLLM(engine="openai", private_key="sk-your-key")

with open("speech.wav", "rb") as f:
    request = AudioTranscriptionsRequest(
        file=f.read(),
        filename="speech.wav",
        content_type="audio/wav",
        model="whisper-1",
        response_format="json",  # json, verbose_json, text, srt, vtt
    )

result = client.llm.sync_audio_transcriptions(request)
print(result["text"])
```

Async:

```python
result = await client.llm.async_audio_transcriptions(request)
```

Validation happens before upload for empty files, oversized files, unsupported response formats, unknown metadata, missing provider-required fields, and Azure language/content-type errors.

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

audio_bytes = await client.llm.async_audio_speech(request)
```

Use sync TTS only for providers marked sync-supported in the matrix.

## Azure Speech

Construct Azure through the public facade with `speech_key` and `speech_region`:

```python
client = MagicLLM(engine="azure", speech_key="azure-speech-key", speech_region="eastus")
```

`private_key` is accepted as a `speech_key` alias only when `speech_key` is absent. Azure TTS escapes text inside SSML, validates voice/language, and maps `response_format` values (`mp3`, `wav`, `pcm`, `ogg`) to Azure output formats. Azure STT requires a valid language such as `en-US` and accepts WAV uploads only; non-WAV audio is rejected instead of relabeled.

## Retry and billing

Supported media calls use bounded retries for transient transport/HTTP failures such as `408`, `429`, and `5xx`. Unsupported-operation, validation, credential/configuration, and non-transient client errors are not retried.

Retries can repeat billable provider work if a provider processes a request but the client observes a timeout. Reduce retry pressure by constructing the client with fewer retries when needed, for example `MagicLLM(..., retries=1)`.

## Image generation boundary

Audio support is separate from vision/image input. Magic LLM does not currently expose a first-class image generation/text-to-image API.
