import json
import time

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatResponse import Choice, Message
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class ProviderAmazonNova(AmazonBaseProvider):
    """
    Provider for Amazon Bedrock Nova models.
    """

    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for Nova models.

        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request

        Returns:
            A JSON string containing the request body
        """
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

    def process_response(self, r: dict) -> ModelChatResponse:
        """
        Process the response from Nova models.

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

        # Map Nova stop reasons to OpenAI format
        stop_reason_map = {
            'end_turn': 'stop',
            'stop_sequence': 'stop',
            'max_tokens': 'length',
            'content_filtered': 'content_filter',
            'tool_use': 'tool_calls'
        }
        finish_reason = stop_reason_map.get(r.get('stopReason', 'end_turn'), 'stop')

        # Create message
        message = Message(
            role=output_message.get('role', 'assistant'),
            content=content,
            tool_calls=tool_calls,
            refusal=None,
            annotations=[]
        )

        # Create choice
        choice = Choice(
            index=0,
            message=message,
            logprobs=None,
            finish_reason=finish_reason
        )

        # Create usage model
        usage_data = r.get('usage', {})
        usage = UsageModel(
            prompt_tokens=usage_data.get('inputTokens', 0),
            completion_tokens=usage_data.get('outputTokens', 0),
            total_tokens=usage_data.get('totalTokens', 0)
        )

        # Create response
        return ModelChatResponse(
            id=f"nova_{int(time.time() * 1000)}",
            object='chat.completion',
            created=int(time.time()),
            model='amazon.nova',  # Could be more specific if model info is available
            choices=[choice],
            usage=usage,
            service_tier=None,
            system_fingerprint=None
        )

    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event from Nova models to a ChatCompletionModel.

        Args:
            event: The event from the streaming response

        Returns:
            A ChatCompletionModel containing the formatted event
        """
        chunk = {
            'id': '1',
            'choices':
                [{
                    'delta':
                        {
                            'content': event.get('contentBlockDelta', {}).get('delta', {}).get('text'),
                            'role': None
                        },
                    'finish_reason': 'stop' if event.get('messageStop', {}).get(
                        'stopReason') == 'end_turn' else None,
                    'index': event.get('index')
                }],
            'created': int(time.time()),
            'model': self.model,
            'object': 'chat.completion.chunk'
        }

        if c := event.get('metadata', {}).get('usage'):
            chunk['usage'] = UsageModel(
                prompt_tokens=c['inputTokens'],
                completion_tokens=c['outputTokens'],
                total_tokens=c['inputTokens'] + c['outputTokens']
            )

        return ChatCompletionModel(**chunk)
