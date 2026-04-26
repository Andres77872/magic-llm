import base64
import pytest

from magic_llm.model import ModelChat
from magic_llm.exception.ChatException import ChatException
from magic_llm.engine.openai_adapters.base_provider import (
    OpenAiBaseProvider, 
    _has_image_content
)
from magic_llm.engine.amazon_adapters.base_provider import (
    AmazonBaseProvider,
)
from magic_llm.engine.amazon_adapters.amazon_meta import ProviderAmazonMeta
from magic_llm.engine.amazon_adapters.amazon_nova import ProviderAmazonNova

PNG_1x1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)
LARGE_B64 = "A" * 10000


class TestTokenCountingMultimodal:
    def test_string_content_counts_tokens(self):
        chat = ModelChat()
        chat.add_user_message("Hello world")
        tokens = chat.num_tokens_from_messages()
        assert tokens > 0

    def test_image_content_estimates_tokens(self):
        chat = ModelChat()
        data_uri = f"data:image/png;base64,{LARGE_B64}"
        chat.add_user_message("Look at this", image=data_uri)
        tokens = chat.num_tokens_from_messages()
        assert tokens > 2500

    def test_multimodal_content_combined_tokens(self):
        chat = ModelChat()
        data_uri = f"data:image/png;base64,{LARGE_B64}"
        chat.add_user_message("Short text", image=data_uri)
        tokens = chat.num_tokens_from_messages()
        assert tokens > 2500

    def test_http_url_image_low_estimate(self):
        chat = ModelChat()
        chat.add_user_message("Check this", image="https://example.com/image.png")
        tokens = chat.num_tokens_from_messages()
        assert tokens > 0
        assert tokens < 200

    def test_truncation_respects_multimodal_budget(self):
        chat = ModelChat(max_input_tokens=500)
        data_uri = f"data:image/png;base64,{LARGE_B64}"
        chat.add_user_message("A message with a large image that should trigger truncation logic", image=data_uri)
        chat.add_user_message("Another message")
        messages = chat.get_messages()
        total = chat.num_tokens_from_messages(messages)
        assert total <= 500

    def test_multiple_images_tokens_counted(self):
        chat = ModelChat()
        chat.add_user_message(
            "Multiple images",
            image=[
                f"data:image/png;base64,{LARGE_B64}",
                f"data:image/png;base64,{LARGE_B64}"
            ]
        )
        tokens = chat.num_tokens_from_messages()
        assert tokens > 5000


class TestVisionCapabilityCheck:
    def test_has_image_content_detects_list_content(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]}
        ]
        assert _has_image_content(messages) is True

    def test_has_image_content_false_for_string(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _has_image_content(messages) is False

    def test_has_image_content_false_for_text_only_list(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        assert _has_image_content(messages) is False

    def test_provider_supports_vision_default_true(self):
        assert OpenAiBaseProvider.supports_vision is True

    def test_nova_supports_vision(self):
        assert ProviderAmazonNova.supports_vision is True

    def test_meta_does_not_support_vision(self):
        assert ProviderAmazonMeta.supports_vision is False


class VisionUnsupportedProvider(OpenAiBaseProvider):
    supports_vision = False
    
    def __init__(self, **kwargs):
        super().__init__(base_url="https://test.example.com/v1", api_key="test", **kwargs)


class TestVisionValidation:
    def test_vision_supported_provider_passes(self):
        class VisionProvider(OpenAiBaseProvider):
            supports_vision = True
            def __init__(self, **kwargs):
                super().__init__(base_url="https://test.example.com/v1", api_key="test", **kwargs)
        
        provider = VisionProvider(model="test-model")
        chat = ModelChat()
        chat.add_user_message("Describe", image=f"data:image/png;base64,{PNG_1x1_BASE64}")
        
        data, headers = provider.transform_request(chat)
        assert data is not None

    def test_vision_unsupported_provider_raises(self):
        provider = VisionUnsupportedProvider(model="test-model")
        chat = ModelChat()
        chat.add_user_message("Describe", image=f"data:image/png;base64,{PNG_1x1_BASE64}")
        
        with pytest.raises(ChatException) as exc_info:
            provider.transform_request(chat)
        
        assert exc_info.value.error_code == 'VISION_NOT_SUPPORTED'
        assert 'does not support image' in exc_info.value.message.lower()

    def test_vision_unsupported_provider_passes_without_images(self):
        provider = VisionUnsupportedProvider(model="test-model")
        chat = ModelChat()
        chat.add_user_message("Just text")
        
        data, headers = provider.transform_request(chat)
        assert data is not None


class MockAmazonProviderNoVision(AmazonBaseProvider):
    supports_vision = False
    
    def transform_request(self, chat, **kwargs):
        self._validate_vision_support(chat)
        return '{"prompt": "test"}'
    
    def transform_response(self, response):
        pass
    
    def transform_stream_chunk(self, event):
        pass


class TestAmazonVisionValidation:
    def test_amazon_validate_vision_raises_on_images(self):
        provider = MockAmazonProviderNoVision(model="meta.llama")
        chat = ModelChat()
        chat.add_user_message("Describe", image=f"data:image/png;base64,{PNG_1x1_BASE64}")
        
        with pytest.raises(ChatException) as exc_info:
            provider.transform_request(chat)
        
        assert exc_info.value.error_code == 'VISION_NOT_SUPPORTED'

    def test_amazon_validate_vision_passes_without_images(self):
        provider = MockAmazonProviderNoVision(model="meta.llama")
        chat = ModelChat()
        chat.add_user_message("Just text")
        
        result = provider.transform_request(chat)
        assert result is not None