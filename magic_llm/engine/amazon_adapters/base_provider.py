import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any

from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class AmazonBaseProvider(ABC):
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
        # No boto3/aioboto3 clients — raw HTTP + SigV4 signing is used by EngineAmazon
        # Credentials may be resolved from ambient IAM chain at request time

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

        Note: Image support varies by Amazon Bedrock model. Override this
        method in subclasses to handle images for models that support them.
        """
        pass

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