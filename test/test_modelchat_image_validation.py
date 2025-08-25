import base64
import pytest

from magic_llm.model import ModelChat

# 1x1 transparent PNG (valid base64)
PNG_1x1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)
PNG_1x1_BYTES = base64.b64decode(PNG_1x1_BASE64)


def _get_last_image_url(chat: ModelChat) -> str:
    msg = chat.messages[-1]
    content = msg["content"]
    assert isinstance(content, list)
    # first element is text, second is image
    img_part = next(p for p in content if isinstance(p, dict) and p.get("type") == "image_url")
    return img_part["image_url"]["url"]


def test_data_uri_with_mime_and_base64_ok():
    data_uri = f"data:image/png;base64,{PNG_1x1_BASE64}"
    chat = ModelChat()
    chat.add_user_message("Describe.", image=data_uri)
    url = _get_last_image_url(chat)
    assert url == data_uri


def test_data_uri_missing_base64_raises():
    bad_data_uri = f"data:image/png,{PNG_1x1_BASE64}"
    chat = ModelChat()
    with pytest.raises(ValueError, match="must declare base64"):
        chat.add_user_message("Describe.", image=bad_data_uri)


def test_data_uri_missing_mime_raises():
    bad_data_uri = f"data:;base64,{PNG_1x1_BASE64}"
    chat = ModelChat()
    with pytest.raises(ValueError, match="must include a MIME type"):
        chat.add_user_message("Describe.", image=bad_data_uri)


def test_raw_base64_without_media_type_raises():
    chat = ModelChat()
    with pytest.raises(ValueError, match="Raw base64 image provided without a valid media_type"):
        chat.add_user_message("Describe.", image=PNG_1x1_BASE64, media_type=None)  # type: ignore[arg-type]


def test_raw_base64_with_media_type_ok():
    chat = ModelChat()
    chat.add_user_message("Describe.", image=PNG_1x1_BASE64, media_type="image/png")
    url = _get_last_image_url(chat)
    assert url.startswith("data:image/png;base64,")
    assert url.split(",", 1)[1] == PNG_1x1_BASE64


def test_bytes_without_media_type_raises():
    chat = ModelChat()
    with pytest.raises(ValueError, match="Bytes image requires a valid media_type"):
        chat.add_user_message("Describe.", image=PNG_1x1_BYTES, media_type=None)  # type: ignore[arg-type]


def test_bytes_with_media_type_ok():
    chat = ModelChat()
    chat.add_user_message("Describe.", image=PNG_1x1_BYTES, media_type="image/png")
    url = _get_last_image_url(chat)
    assert url.startswith("data:image/png;base64,")
    # ensure the payload matches the encoded bytes
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == PNG_1x1_BYTES
