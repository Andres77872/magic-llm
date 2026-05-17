import json
import time

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.engine.tooling import guard_tools_supported
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_llm.util.response_mapping import (
    AMAZON_FINISH_REASON_MAP,
    map_finish_reason,
    build_response,
    build_stream_chunk,
)


class ProviderAmazonNova(AmazonBaseProvider):
    """
    Provider for Amazon Bedrock Nova models.
    """
    supports_vision: bool = True

    def transform_request(self, chat: ModelChat, **kwargs) -> str:
        """
        Transform ModelChat to Nova request format.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body

        Note: Nova models support images via the content array format.
        Image support can be added by including image parts in messages.
        """
        guard_tools_supported('Amazon Bedrock Nova', kwargs.get('tools'), kwargs.get('tool_choice'))
        m = chat.get_messages()
        for i in m:
            if (c := i.get('content')) and type(c) == str:
                i['content'] = [{
                    "text": i['content']
                }]

        body = json.dumps({
            "messages": m,
            "inferenceConfig": {
                "max_new_tokens": kwargs.get('max_new_tokens', 4096),
                "temperature": kwargs.get('temperature', 1),
                "topP": kwargs.get('topP', 1)
            }
        })

        return body

    def transform_response(self, r: dict) -> ModelChatResponse:
        """
        Transform Nova response to ModelChatResponse.

        Args:
            r: The response from the model

        Returns:
            A ModelChatResponse object
        """
        # Extract content and tool calls from Nova response
        output_message = r['output']['message']
        content_parts = []
        tool_calls = None

        for content_item in output_message['content']:
            if 'text' in content_item:
                content_parts.append(content_item['text'])

        # Combine text parts
        content = ''.join(content_parts) if content_parts else None

        # Map Nova stop reasons to OpenAI format using shared mapping
        finish_reason = map_finish_reason(
            r.get('stopReason', 'end_turn'),
            AMAZON_FINISH_REASON_MAP,
            default='stop'
        )

        # Create usage model
        usage_data = r.get('usage', {})
        usage = UsageModel(
            prompt_tokens=usage_data.get('inputTokens', 0),
            completion_tokens=usage_data.get('outputTokens', 0),
            total_tokens=usage_data.get('totalTokens', 0)
        )

        # Build standardized response
        return build_response(
            id=f"nova_{int(time.time() * 1000)}",
            model='amazon.nova',
            content=content,
            role=output_message.get('role', 'assistant'),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage
        )

    def transform_stream_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Transform streaming event from Nova models to ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        content = event.get('contentBlockDelta', {}).get('delta', {}).get('text')
        stop_reason = event.get('messageStop', {}).get('stopReason')
        finish_reason = map_finish_reason(stop_reason, AMAZON_FINISH_REASON_MAP) if stop_reason else None

        usage = None
        if c := event.get('metadata', {}).get('usage'):
            usage = UsageModel(
                prompt_tokens=c['inputTokens'],
                completion_tokens=c['outputTokens'],
                total_tokens=c['inputTokens'] + c['outputTokens']
            )

        return build_stream_chunk(
            id='1',
            model=self.model,
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            index=event.get('index', 0)
        )
