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
