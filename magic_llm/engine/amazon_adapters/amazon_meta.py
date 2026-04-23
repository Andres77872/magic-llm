import json
import time

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.response_mapping import (
    build_response,
    build_stream_chunk,
)


class ProviderAmazonMeta(AmazonBaseProvider):
    """
    Provider for Meta Llama models via Amazon Bedrock.
    """

    def transform_request(self, chat: ModelChat, **kwargs) -> str:
        """
        Transform ModelChat to Meta Llama request format.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body

        Raises:
            ChatException: If request contains images

        Note: Meta Llama models do not support image inputs.
        """
        self._validate_vision_support(chat)
        body = json.dumps({
            "prompt": chat.generic_chat(format='llama2'),
            "max_gen_len": kwargs.get('max_gen_len', 1024),
            "temperature": kwargs.get('temperature', 0.2),
            "top_p": kwargs.get('top_p', 1),
        })

        return body

    def transform_response(self, response: dict) -> ModelChatResponse:
        """
        Transform Meta Llama response to ModelChatResponse.

        Args:
            response: The response from the model

        Returns:
            A ModelChatResponse object
        """
        # Create usage model
        usage = UsageModel(
            prompt_tokens=response['prompt_token_count'],
            completion_tokens=response['generation_token_count'],
            total_tokens=response['prompt_token_count'] + response['generation_token_count']
        )

        # Determine finish reason
        stop_reason = response.get('stop_reason')
        finish_reason = 'stop' if stop_reason else None

        # Build standardized response
        return build_response(
            id=f"llama_{int(time.time() * 1000)}",
            model='meta.llama',
            content=response['generation'],
            finish_reason=finish_reason,
            usage=usage
        )

    def transform_stream_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Transform streaming event from Meta Llama to ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        finish_reason = 'stop' if event.get('stop_reason') else None

        return build_stream_chunk(
            id='1',
            model=self.model,
            content=event['generation'],
            finish_reason=finish_reason
        )