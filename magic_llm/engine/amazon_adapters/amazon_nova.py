import json
import time
from typing import Dict

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat
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

    def process_response(self, response: dict) -> Dict:
        """
        Process the response from Nova models.

        Args:
            response: The response from the model

        Returns:
            A dictionary containing the processed response
        """
        u = response.get('usage', {})
        return {
            'content': response['output']['message']['content'][0]['text'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=u['inputTokens'],
                completion_tokens=u['outputTokens'],
                total_tokens=u['totalTokens']
            )
        }

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
