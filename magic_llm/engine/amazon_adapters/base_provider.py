import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

from magic_llm.exception.ChatException import ChatException
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


def _has_image_content(messages: list[dict]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    return True
    return False


class AmazonBaseProvider(ABC):
    supports_vision: bool = False
    
    def __init__(self,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 model: str | None = None,
                 **kwargs):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.service_name = service_name
        self.model = model
        self.kwargs = kwargs

    # ═══════════════════════════════════════════════════════════════════
    # TRANSFORMATION METHODS (Provider-specific, must be implemented)
    # ═══════════════════════════════════════════════════════════════════

    @abstractmethod
    def transform_request(self, chat: ModelChat, **kwargs) -> str:
        """
        Transform ModelChat to provider-specific JSON string.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body

        Raises:
            ChatException: If request contains images but provider doesn't support vision

        Note: Image support varies by Amazon Bedrock model. Override this
        method in subclasses to handle images for models that support them.
        Subclasses should call _validate_vision_support(chat) before processing.
        """
        pass

    def _validate_vision_support(self, chat: ModelChat) -> None:
        """
        Validate that provider supports images if the chat contains them.
        
        Raises:
            ChatException: If request contains images but provider doesn't support vision
        """
        messages = chat.get_messages()
        if _has_image_content(messages) and not self.supports_vision:
            raise ChatException(
                message=f"Provider '{self.__class__.__name__}' does not support image/vision inputs. "
                        f"Model '{self.model}' cannot process images. Remove images or use a vision-capable model.",
                error_code='VISION_NOT_SUPPORTED'
            )

    @abstractmethod
    def transform_response(self, response: dict) -> ModelChatResponse:
        """
        Transform provider response to ModelChatResponse.

        IMPORTANT: Must return ModelChatResponse, not dict.

        Args:
            response: The response from the model

        Returns:
            A ModelChatResponse object
        """
        pass

    @abstractmethod
    def transform_stream_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Transform streaming event to ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        pass

    # ═══════════════════════════════════════════════════════════════════
    # BACKWARD COMPATIBLE ALIASES
    # ═══════════════════════════════════════════════════════════════════

    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for the model.
        Alias for transform_request for backward compatibility.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body
        """
        return self.transform_request(chat, **kwargs)

    def process_response(self, response: dict) -> ModelChatResponse:
        """
        Process the response from the model.
        Alias for transform_response for backward compatibility.

        IMPORTANT: Returns ModelChatResponse, not dict.

        Args:
            response: The response from the model

        Returns:
            A ModelChatResponse object
        """
        return self.transform_response(response)

    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event to a ChatCompletionModel.
        Alias for transform_stream_chunk for backward compatibility.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        return self.transform_stream_chunk(event)