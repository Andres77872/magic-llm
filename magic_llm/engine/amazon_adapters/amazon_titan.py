import json
import time

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.response_mapping import (
    AMAZON_FINISH_REASON_MAP,
    map_finish_reason,
    build_response,
    build_stream_chunk,
)


class ProviderAmazonTitan(AmazonBaseProvider):
    """
    Provider for Amazon Bedrock Titan models.
    """

    def transform_request(self, chat: ModelChat, **kwargs) -> str:
        """
        Transform ModelChat to Titan request format.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body

        Note: Titan models do not support image inputs.
        """
        body = json.dumps({
            "inputText": chat.generic_chat(format='titan'),
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get('maxTokenCount', 4096),
                "temperature": kwargs.get('temperature', 0),
                "topP": kwargs.get('topP', 1),
                "stopSequences": kwargs.get('stopSequences', ['User:']),
            }
        })

        return body

    def transform_response(self, response: dict) -> ModelChatResponse:
        """
        Transform Titan response to ModelChatResponse.

        Args:
            response: The response from the model

        Returns:
            A ModelChatResponse object
        """
        # Create usage model
        usage = UsageModel(
            prompt_tokens=response['inputTextTokenCount'],
            completion_tokens=response['results'][0]['tokenCount'],
            total_tokens=response['inputTextTokenCount'] + response['results'][0]['tokenCount'],
        )

        # Map completion reason to finish reason
        completion_reason = response['results'][0].get('completionReason', 'FINISH')
        finish_reason = map_finish_reason(
            completion_reason,
            AMAZON_FINISH_REASON_MAP,
            default='stop'
        )

        # Build standardized response
        return build_response(
            id=f"titan_{int(time.time() * 1000)}",
            model='amazon.titan',
            content=response['results'][0]['outputText'],
            finish_reason=finish_reason,
            usage=usage
        )

    def transform_stream_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Transform streaming event from Titan models to ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        finish_reason = map_finish_reason(
            event.get('completionReason'),
            AMAZON_FINISH_REASON_MAP
        ) if event.get('completionReason') else None

        return build_stream_chunk(
            id='1',
            model=self.model,
            content=event['outputText'],
            finish_reason=finish_reason,
            index=event.get('index', 0)
        )