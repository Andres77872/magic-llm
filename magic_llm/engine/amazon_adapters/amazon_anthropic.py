import json
import time

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.response_mapping import (
    ANTHROPIC_FINISH_REASON_MAP,
    map_finish_reason,
    build_response,
    build_stream_chunk,
)


class ProviderAmazonAnthropic(AmazonBaseProvider):
    """
    Provider for Anthropic Claude models via Amazon Bedrock.
    """

    def transform_request(self, chat: ModelChat, **kwargs) -> str:
        """
        Transform ModelChat to Anthropic Claude (Bedrock) request format.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body

        Note: This is for legacy Claude format via Bedrock. Image support
        would require updating to the Messages API format.
        """
        body = json.dumps({
            "prompt": chat.generic_chat(format='claude'),
            "max_tokens_to_sample": kwargs.get('max_tokens_to_sample', 1024),
            "temperature": kwargs.get('temperature', 0.5),
            "top_k": kwargs.get('top_k', 250),
            "top_p": kwargs.get('top_p', 1),
            "stop_sequences": kwargs.get('stop_sequences', ["\n\nHuman:"]),
            # "anthropic_version": "bedrock-2023-05-31"
        })

        return body

    def transform_response(self, response: dict) -> ModelChatResponse:
        """
        Transform Anthropic Claude (Bedrock) response to ModelChatResponse.

        Args:
            response: The response from the model

        Returns:
            A ModelChatResponse object
        """
        # Create usage model (approximate tokens from character count)
        usage = UsageModel(
            prompt_tokens=len(response.get('prompt', '')),
            completion_tokens=len(response['completion']),
            total_tokens=len(response.get('prompt', '')) + len(response['completion'])
        )

        # Map stop_reason to finish_reason
        finish_reason = map_finish_reason(
            response.get('stop_reason'),
            ANTHROPIC_FINISH_REASON_MAP,
            default='stop'
        )

        # Build standardized response
        return build_response(
            id=f"bedrock_claude_{int(time.time() * 1000)}",
            model='anthropic.claude',
            content=response['completion'],
            finish_reason=finish_reason,
            usage=usage
        )

    def transform_stream_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Transform streaming event from Anthropic Claude to ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        finish_reason = 'stop' if event.get('stop_reason') else None

        return build_stream_chunk(
            id='1',
            model=self.model,
            content=event['completion'],
            finish_reason=finish_reason
        )