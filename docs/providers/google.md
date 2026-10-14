# Google AI Studio

Use `engine="google"` for the native Gemini Developer API with an API key
created in Google AI Studio. The adapter authenticates using `x-goog-api-key`.

```python
from magic_llm import MagicLLM
import os

client = MagicLLM(
    engine="google",
    model="gemini-3.8-flash",
    private_key=os.environ["GEMINI_API_KEY"],
    # Optional transport settings; neither is sent in the JSON body.
    base_url="https://generativelanguage.googleapis.com/v1beta",
    timeout=60,
    temperature=1,
    top_p=0.9,
    max_tokens=1024,
)
```

## Request options

The native endpoints are `/models/{model}:generateContent` and
`/models/{model}:streamGenerateContent?alt=sse`. Omit `base_url` to use the
default above. Bare API hosts and roots ending in `/models` are also accepted.
Model names may include the `models/` prefix.

Generation options go in `generationConfig`. Native names such as `topP`,
`maxOutputTokens`, `responseMimeType`, and `thinkingConfig` are preserved.
Common aliases are translated:

| Client option | Gemini field |
| --- | --- |
| `top_p`, `top_k` | `topP`, `topK` |
| `max_tokens`, `max_completion_tokens`, `max_output_tokens` | `maxOutputTokens` |
| `stop`, `stop_sequences` | `stopSequences` (a string becomes a one-item list) |
| `presence_penalty`, `frequency_penalty` | `presencePenalty`, `frequencyPenalty` |

`safetySettings`/`safety_settings` and `cachedContent`/`cached_content` are sent
at the request's top level. Tools and tool choice are mapped separately.
Call-time generation options override constructor defaults, including across
aliases; `None` clears an optional default. A call-time `timeout` overrides
the constructor timeout for the HTTP request. Configure `base_url` when
constructing the client.

## OpenAI-compatible access

Google also exposes an OpenAI-compatible API. To use that protocol, choose
`engine="openai"` with
`base_url="https://generativelanguage.googleapis.com/v1beta/openai/"` and
the same Gemini API key. The native Google engine rejects `/openai` roots
with a configuration error so the two protocols cannot be mixed accidentally.

## Notes

- Chat, streaming, async chat, and async streaming are supported.
- Tool schemas are mapped into Gemini's expected format.
- Model discovery is supported.
- Vision inputs are supported through `ModelChat.add_user_message(..., image=...)` where the model supports images.
- Gemini TTS is supported through both `audio_speech(...)` and `async_audio_speech(...)`; returned audio is WAV bytes.
- STT/transcription is unsupported for the Google engine in core.

## Example

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Be brief.")
chat.add_user_message("Summarize why streaming responses help UX.")

for chunk in client.llm.stream_generate(chat):
    print(chunk.choices[0].delta.content or "", end="")
```

## References

- [Native content generation and request schema](https://ai.google.dev/api/generate-content)
- [Gemini API authentication and endpoints](https://ai.google.dev/gemini-api/docs/api-overview)
- [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
