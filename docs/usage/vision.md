# Vision

Vision input is built with `ModelChat.add_user_message(content, image=...)`. The model and provider must support image input.

## Image URL

```python
from magic_llm.model import ModelChat

chat = ModelChat(system="Describe images accurately.")
chat.add_user_message(
    "What is in this image?",
    image="https://example.com/image.jpg",
)
```

## Image bytes

```python
with open("diagram.png", "rb") as f:
    image_bytes = f.read()

chat.add_user_message(
    "Summarize this diagram.",
    image=image_bytes,
    media_type="image/png",
)
```

## Raw base64

```python
chat.add_user_message(
    "What changed in this screenshot?",
    image="iVBORw0KGgoAAAANSUhEUgAA...",
    media_type="image/png",
)
```

## Data URI

```python
chat.add_user_message(
    "Extract the visible text.",
    image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
)
```

## Multiple images

```python
chat.add_user_message(
    "Compare these two images.",
    image=["https://example.com/a.jpg", "https://example.com/b.jpg"],
)
```

## Validation rules

- Image-only messages are rejected; include text content.
- Bytes and raw base64 require a valid `media_type`, for example `image/png`.
- Data URIs must include a MIME type and `;base64`.

Vision support depends on the selected provider and model. If uncertain, test with the provider's known vision-capable model.

## Vision input is not image generation

This page documents **image input** for chat/vision models. Magic LLM does not currently provide a first-class image generation/text-to-image API, `ModelImage`, or `/images/generations` proxy. A user-defined tool named `generate_image` is just a caller-owned tool name, not a core media feature.

## Provider support

Image input is accepted by official OpenAI (Chat Completions and `endpoint="responses"`), OpenRouter, Anthropic, and Google. Every other adapter rejects images before a provider request: Groq, SambaNova, Mistral, Fireworks, DeepSeek, DeepInfra, Together, generic OpenAI-compatible endpoints, all Amazon Bedrock adapters, Cloudflare, and Cohere.

The rejection is a `VISION_NOT_SUPPORTED` error, and the request is not retried. With a `fallback` client configured, the fallback receives the same chat, so a vision-capable fallback can answer it. Without one, the call raises `ChatException` with `error_code="STREAM_GENERATION_ERROR"`, keeping the original error as `__context__`. Amazon Bedrock vision stays disabled until a native Bedrock image payload transform is implemented and tested.
