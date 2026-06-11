# Google AI Studio

Use `engine="google"` for Gemini models through Google AI Studio.

```python
from magic_llm import MagicLLM

client = MagicLLM(
    engine="google",
    model="gemini-1.5-flash",
    private_key="google-api-key",
)
```

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
